// Package neurofly runs a trained neurofly controller from Go: frames and sound in,
// keyboard and mouse controls out.
//
// It spawns `neurofly-core serve <artifact>` and speaks the JSON-lines protocol in
// artifact/SPEC.md. The runtime must be installed in a Python on the PATH
// (`pip install neurofly-core`), or use WithPython to name an interpreter.
//
//	fly, err := neurofly.Spawn("artifacts/myapp")
//	r, err := fly.Step(neurofly.Step{Frame: rgb, Width: 320, Height: 240})
//	// r.Keys, r.Buttons, r.DX, r.DY, r.Scroll, r.Held, r.Action, r.Spikes
//	fly.Close()
//
// Training from Go: start from a base artifact (`neurofly build`), read features with
// Observe, train a policy, install it with SetPolicyLinear or SetPolicyMLP, and Save.
package neurofly

import (
	"bufio"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
)

// Info is what the runtime announces on start.
type Info struct {
	Name        string   `json:"name"`
	NNeurons    int      `json:"n_neurons"`
	NFeatures   int      `json:"n_features"`
	NActions    int      `json:"n_actions"`
	Controls    []string `json:"controls"`
	BrainMs     float64  `json:"brain_ms"`
	HasPolicy   bool     `json:"has_policy"`
	HasAudition bool     `json:"has_audition"`
	SampleRate  *int     `json:"sample_rate"`
}

// Step is one step's input. Frame is raw RGB bytes (row-major, three per pixel) of
// Width x Height, or an encoded image when Format is "png" or "jpeg". Audio is the
// samples since the last step in [-1, 1], interleaved if Channels > 1.
type Step struct {
	Frame      []byte
	Width      int
	Height     int
	Format     string
	Audio      []float32
	SampleRate int
	Channels   int
	Reward     *float64
	Detections []Detection // objects a detector found, for a detection encoder
}

// Detection is one detected object: a class id (index into the artifact's detection
// classes), a box in fractions of the frame (left, top, right, bottom) and a score in [0, 1].
type Detection struct {
	Class int        `json:"class"`
	Box   [4]float32 `json:"box"`
	Score float32    `json:"score"`
}

// Result is the runtime's answer to Step (controls) or Observe (Features).
type Result struct {
	T        int       `json:"t"`
	Spikes   int       `json:"spikes"`
	Action   []float64 `json:"action"`
	Held     []string  `json:"held"`
	Keys     []string  `json:"keys"`
	Buttons  []string  `json:"buttons"`
	DX       float64   `json:"dx"`
	DY       float64   `json:"dy"`
	Scroll   float64   `json:"scroll"`
	Features []float64 `json:"features"`
	Activity *Activity `json:"activity,omitempty"`
}

// Activity is every neuron that fired during one step (after Client.Activity(true)).
type Activity struct {
	T       int     `json:"t"`
	Indices []int64 `json:"indices"`
	Counts  []int   `json:"counts"`
}

// BrainMap is where every neuron is, for drawing the brain.
type BrainMap struct {
	N           int                `json:"n"`
	Unit        string             `json:"unit"`
	Positions   [][]float32        `json:"positions"`
	Known       []bool             `json:"known"`
	Superclass  []string           `json:"superclass"`
	Populations map[string][]int64 `json:"populations"`
}

// Layer is one dense layer of an MLP policy: W is out x in.
type Layer struct {
	W [][]float64 `json:"W"`
	B []float64   `json:"b"`
}

// Client is a running neurofly-core process.
type Client struct {
	cmd   *exec.Cmd
	stdin io.WriteCloser
	out   *bufio.Reader
	Info  Info
}

type envelope struct {
	OK    bool   `json:"ok"`
	Error string `json:"error"`
	Bye   bool   `json:"bye"`
	Path  string `json:"path"`
}

// Spawn starts `neurofly-core serve artifact`.
func Spawn(artifact string) (*Client, error) {
	return start(exec.Command("neurofly-core", "serve", artifact))
}

// WithPython starts `python -m neurofly_core serve artifact`.
func WithPython(python, artifact string) (*Client, error) {
	return start(exec.Command(python, "-m", "neurofly_core", "serve", artifact))
}

func start(cmd *exec.Cmd) (*Client, error) {
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	c := &Client{cmd: cmd, stdin: stdin, out: bufio.NewReaderSize(stdout, 1<<20)}
	line, err := c.readLine()
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(line, &c.Info); err != nil {
		return nil, err
	}
	return c, nil
}

func (c *Client) readLine() ([]byte, error) {
	line, err := c.out.ReadBytes('\n')
	if err != nil && len(line) == 0 {
		return nil, fmt.Errorf("neurofly-core exited: %w", err)
	}
	var env envelope
	if err := json.Unmarshal(line, &env); err != nil {
		return nil, err
	}
	if !env.OK {
		return nil, errors.New("neurofly-core: " + env.Error)
	}
	return line, nil
}

func (c *Client) call(req map[string]any, out any) error {
	b, err := json.Marshal(req)
	if err != nil {
		return err
	}
	if _, err := c.stdin.Write(append(b, '\n')); err != nil {
		return err
	}
	line, err := c.readLine()
	if err != nil {
		return err
	}
	if out != nil {
		return json.Unmarshal(line, out)
	}
	return nil
}

func (c *Client) stepRequest(op string, s Step) map[string]any {
	req := map[string]any{"op": op, "frame": base64.StdEncoding.EncodeToString(s.Frame),
		"width": s.Width, "height": s.Height}
	if s.Format != "" {
		req["format"] = s.Format
	}
	if len(s.Audio) > 0 {
		buf := make([]byte, 4*len(s.Audio))
		for i, x := range s.Audio {
			binary.LittleEndian.PutUint32(buf[4*i:], math.Float32bits(x))
		}
		req["audio"] = base64.StdEncoding.EncodeToString(buf)
		sr, ch := s.SampleRate, s.Channels
		if sr == 0 {
			sr = 16000
		}
		if ch == 0 {
			ch = 1
		}
		req["sample_rate"], req["channels"] = sr, ch
	}
	if s.Reward != nil {
		req["reward"] = *s.Reward
	}
	if len(s.Detections) > 0 {
		req["detections"] = s.Detections
	}
	return req
}

// Reset starts a fresh episode.
func (c *Client) Reset() error { return c.call(map[string]any{"op": "reset"}, nil) }

// Step feeds a frame (and sound) and returns the controls.
func (c *Client) Step(s Step) (*Result, error) {
	var r Result
	return &r, c.call(c.stepRequest("step", s), &r)
}

// Observe feeds a frame (and sound) and returns the feature vector, no policy involved.
func (c *Client) Observe(s Step) (*Result, error) {
	var r Result
	return &r, c.call(c.stepRequest("observe", s), &r)
}

// SetPolicyLinear installs action = tanh(W f + b); W is actions x features.
func (c *Client) SetPolicyLinear(w [][]float64, b []float64) error {
	return c.call(map[string]any{"op": "set_policy", "type": "linear", "W": w, "b": b}, nil)
}

// SetPolicyMLP installs dense layers with the given activation ("tanh" or "relu") between
// them; obsMean and obsVar (may be nil) normalise the features first.
func (c *Client) SetPolicyMLP(layers []Layer, activation string, obsMean, obsVar []float64) error {
	req := map[string]any{"op": "set_policy", "type": "mlp", "layers": layers, "activation": activation}
	if obsMean != nil {
		req["obs_mean"], req["obs_var"] = obsMean, obsVar
	}
	return c.call(req, nil)
}

// Save writes the runtime's current model (with the installed policy) as an artifact.
// Activity records every neuron's spikes: later Step / Observe results carry them in
// Result.Activity. It returns the neuron count.
func (c *Client) Activity(on bool) (int, error) {
	var r struct{ N int `json:"n"` }
	err := c.call(map[string]any{"op": "activity", "on": on}, &r)
	return r.N, err
}

// Positions returns every neuron's position (micrometres) and superclass, for a viewer.
func (c *Client) Positions() (*BrainMap, error) {
	var m BrainMap
	if err := c.call(map[string]any{"op": "positions"}, &m); err != nil {
		return nil, err
	}
	return &m, nil
}

func (c *Client) Save(dir, name string) (string, error) {
	var env envelope
	req := map[string]any{"op": "save", "path": dir}
	if name != "" {
		req["name"] = name
	}
	return env.Path, c.call(req, &env)
}

// Close stops the runtime; every key and button it held is released.
func (c *Client) Close() error {
	_ = c.call(map[string]any{"op": "close"}, nil)
	_ = c.stdin.Close()
	return c.cmd.Wait()
}
