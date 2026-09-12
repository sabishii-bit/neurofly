// A Go program that uses a base artifact: reads features, installs a policy, saves the
// result, and then gets controls from it.
//
//	neurofly build artifacts/base --brain synthetic --keys w,a --mouse    # once, in Python
//	cd bindings/go && go run ./example ../../artifacts/base ../../artifacts/from_go
package main

import (
	"fmt"
	"math"
	"os"

	"neurofly"
)

func main() {
	if len(os.Args) < 3 {
		fmt.Println("usage: go run ./example <base artifact> <out artifact>")
		os.Exit(2)
	}
	fly, err := neurofly.Spawn(os.Args[1])
	if err != nil {
		panic(err)
	}
	defer fly.Close()
	info := fly.Info
	fmt.Printf("%s: %d neurons, %d features, controls %v\n", info.Name, info.NNeurons, info.NFeatures, info.Controls)

	w, h := 64, 48
	frame := make([]byte, w*h*3)
	audio := make([]float32, 1600)
	for i := range audio {
		audio[i] = 0.2 * float32(math.Sin(float64(i)*0.1))
	}
	for t := 0; t < 3; t++ {
		for i := range frame {
			frame[i] = byte((i*7 + t*31) & 255)
		}
		r, err := fly.Observe(neurofly.Step{Frame: frame, Width: w, Height: h, Audio: audio, SampleRate: 16000})
		if err != nil {
			panic(err)
		}
		fmt.Printf("t=%d features=%d spikes=%d\n", r.T, len(r.Features), r.Spikes)
	}

	// A policy you trained: here a table that always holds the first key.
	W := make([][]float64, info.NActions)
	b := make([]float64, info.NActions)
	for a := range W {
		W[a] = make([]float64, info.NFeatures)
		b[a] = -1
	}
	b[0] = 1
	if err := fly.SetPolicyLinear(W, b); err != nil {
		panic(err)
	}
	r, err := fly.Step(neurofly.Step{Frame: frame, Width: w, Height: h})
	if err != nil {
		panic(err)
	}
	fmt.Printf("with the policy: holds %v, mouse %.1f,%.1f\n", r.Keys, r.DX, r.DY)
	path, err := fly.Save(os.Args[2], "from_go")
	if err != nil {
		panic(err)
	}
	fmt.Println("saved", path)
}
