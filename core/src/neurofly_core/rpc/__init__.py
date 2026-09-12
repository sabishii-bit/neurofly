"""The runtime as a gRPC service (``neurofly.proto``; generated ``neurofly_pb2*`` modules).

    neurofly-core serve artifacts/myapp --grpc 127.0.0.1:50051

Frames and audio travel as bytes, and ``Stream`` keeps one connection open for a
low-latency loop. Regenerate the Python stubs after editing the proto:

    python -m grpc_tools.protoc -I core/src/neurofly_core/rpc \\
        --python_out=core/src/neurofly_core/rpc --grpc_python_out=core/src/neurofly_core/rpc \\
        core/src/neurofly_core/rpc/neurofly.proto

and fix the import in ``neurofly_pb2_grpc.py`` to ``from neurofly_core.rpc import neurofly_pb2``.
"""
