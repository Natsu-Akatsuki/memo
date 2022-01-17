import onnx

def print_shape_info(channel):
    for input in eval(f"model.graph.{channel}"):
        print(input.name, end=": ")
        # get type of input tensor
        tensor_type = input.type.tensor_type
        # check if it has a shape:
        if tensor_type.HasField("shape"):
            # iterate through dimensions of the shape:
            for d in tensor_type.shape.dim:
                # the dimension may have a definite (integer) value or a symbolic identifier or neither:
                if d.HasField("dim_value"):
                    print(d.dim_value, end=", ")  # known dimension
                elif d.HasField("dim_param"):
                    print(d.dim_param, end=", ")  # unknown dimension with symbolic name
                else:
                    print("?", end=", ")  # unknown dimension with no name
        else:
            print("unknown rank", end="")

model_path = "....onnx"
model = onnx.load(model_path)

print_shape_info("input")
print()
print_shape_info("output")

