# bitmap

const RES_PATH = isassigned(ARGS, 1) ? ARGS[1] : "./res"

struct Bitmap
    width::UInt
    height::UInt
    num_channels::UInt8
    max_color::UInt16
    colors::Vector{UInt16}
end

function read_pbm_file(name::String, path::String, verbose::Bool)::Bitmap
    fullpath = abspath(joinpath(path, "$name.pbm"))
    if verbose
        println("path: $fullpath")
    end

    num_channels = max_color = width = height = colors = nothing
    open(fullpath, "r") do file_io
        # header
        magic_number = file_io |> readline |> strip
        @assert startswith(magic_number, "P") "first line must start with 'P'. - $magic_number"
        if magic_number == "P1"
            # 0-1 (white & black)
            num_channels = 1
            max_color = 1
        else
            error("not supported magic number. - P$magic_number")
        end

        # color stream
        width, height = parse.([UInt], split(readline(file_io), " "))
        colors = zeros(Int16, width * height * num_channels)
        color_index = 1
        for line in eachline(file_io)
            for color in parse.([UInt16], split(line, " "))
                colors[color_index] = color
                color_index += 1
            end
        end
    end

    bitmap = Bitmap(width, height, num_channels, max_color, colors)
    if verbose
        println("size: ($width, $height), num of channels: $num_channels, max color: $max_color")
        println("length: ", length(colors))
        println("content:")
        show(bitmap)
    end
    return bitmap
end

function float_colors(bitmap::Bitmap)::Vector{Float32}
    return bitmap.colors ./ bitmap.max_color
end

function pick_color(bitmap::Bitmap, x::UInt, y::UInt)::Vector{UInt16}
    offset = (y-1)*bitmap.width + x*bitmap.num_channels
    limit = offset + bitmap.num_channels - 1
    return bitmap.colors[offset:limit]
end

function pick_color_code(bitmap::Bitmap, x::UInt, y::UInt)::String
    pad = 1
    if bitmap.max_color >= 0xffffff
        pad = 4
    elseif bitmap.max_color >= 0xff
        pad = 2
    end
    return join([string(color, base=16, pad=pad) for color in pick_color(bitmap, x, y)], "")
end

function show(bitmap::Bitmap)
    for y = 1:bitmap.height
        line = join([pick_color_code(bitmap, x, y) for x = 1:bitmap.width], " ")
        println("$line")
    end
end

Bitmap(name::String; path::String=RES_PATH, verbose::Bool=false) = read_pbm_file(name, path, verbose)


# bias

const TRAINING_ALPHA = 0.9f0
const TRAINING_ETA = 0.1f0

mutable struct Bias
    value::Float32
    momentum::Float32

    Bias() = new(zero(Float32), zero(Float32))
end

function random_biases(num::Int)::Vector{Bias}
    biases = fill(Bias(), num)
    for i = 1:num
        biases[i].value = rand() - 0.5  # -0.5 ~ 0.5
        biases[i].momentum = zero(Float32)
    end
    return biases
end

function update!(bias::Bias, delta::Float32)
    delta_bias = TRAINING_ALPHA*bias.momentum + TRAINING_ETA*delta
    bias.value += delta_bias
    bias.momentum = delta_bias
end


# weight

struct Weight
    values::Vector{Float32}
    momentums::Vector{Float32}

    Weight(num::Int) = new(Vector{Float32}(undef, num), zeros(Float32, num))
end

function random_weight(num::Int)::Weight
    weight = Weight(num)
    for i = 1:num
        weight.values[i] = rand() - 0.5 # -0.5 ~ 0.5
    end
    return weight
end

function update!(weight::Weight, delta::Float32, prev_outputs::Vector{Float32})
    for i = 1:length(weight.values)
        delta_weight = TRAINING_ALPHA*weight.momentums[i] + TRAINING_ETA*prev_outputs[i]*delta
        weight.values[i] += delta_weight
        weight.momentums[i] = delta_weight
    end
end


# layer

abstract type Layer end

sigmoid(value::Float32)::Float32 = one(Float32) / (one(Float32) + exp(-value))

function input!(layer::Layer, prev_layer::Layer)
    num_prev_outputs = length(prev_layer.outputs)

    for i = 1:length(layer.outputs)
        if !isassigned(layer.weights, i)
            layer.weights[i] = random_weight(num_prev_outputs)
        end

        sum_nets = zero(Float32)
        for j = 1:num_prev_outputs
            sum_nets += layer.weights[i].values[j] * prev_layer.outputs[j]
        end
        layer.outputs[i] = sigmoid(sum_nets + layer.biases[i].value)
    end
end

struct InputLayer <: Layer
    outputs::Vector{Float32}

    InputLayer(num::Int) = new(zeros(Float32, num))
end

function input!(layer::InputLayer, bitmaps::Bitmap)
    copyto!(layer.outputs, float_colors(bitmaps))
end

struct HiddenLayer <: Layer
    biases::Vector{Bias}
    weights::Vector{Weight}
    deltas::Vector{Float32}
    outputs::Vector{Float32}

    HiddenLayer(num::Int) = new(
        random_biases(num), Vector{Weight}(undef, num), zeros(Float32, num), zeros(Float32, num))
end

function update!(layer::HiddenLayer, prev_layer::Layer, next_layer::Layer)
    copyto!(layer.deltas, next_layer.deltas)

    for i = 1:length(layer.outputs)
        layer.deltas[i] *= layer.outputs[i] * (one(Float32) - layer.outputs[i]) # delta=o*(1-o)*SUM(u_delta*Woh)
        delta = layer.deltas[i]

        update!(layer.biases[i], delta)
        update!(layer.weights[i], delta, prev_layer.outputs)
    end
end

mutable struct OutputLayer <: Layer
    biases::Vector{Bias}
    weights::Vector{Weight}
    deltas::Vector{Float32}
    outputs::Vector{Float32}

    OutputLayer(num::Int) = new(
        random_biases(num), Vector{Weight}(undef, num), zeros(Float32, num), zeros(Float32, num))
end

function diff_target(layer::OutputLayer, index::Int, is_correct::Bool)::Float32
    return (is_correct ? one(Float32) : zero(Float32)) - layer.outputs[index]
end

function update!(layer::OutputLayer, prev_layer::Layer, target_index::Int)::Float64
    num_prev_outputs = length(prev_layer.outputs)
    layer.deltas = zeros(Float32, num_prev_outputs)

    num_outputs = length(layer.outputs)
    sum_errors::Float64 = zero(Float64)
    for i = 1:num_outputs
        diff = diff_target(layer, i, i == target_index)
        sum_errors += 0.5 * diff^2.0                                        # E=1/2(d-o)^2
        delta = diff * layer.outputs[i] * (one(Float32) - layer.outputs[i]) # delta=(d-o)o(1-o)
        for j = 1:num_prev_outputs
            layer.deltas[j] += delta * layer.weights[i].values[j]
        end

        update!(layer.biases[i], delta)
        update!(layer.weights[i], delta, prev_layer.outputs)
    end
    return sum_errors / num_outputs
end


# recognizer

abstract type NeuralNetwork end

struct Recognizer <: NeuralNetwork
    input_layer::InputLayer
    hidden_layer::HiddenLayer
    output_layer::OutputLayer

    Recognizer(input_size::Int, output_size::Int) = new(
        InputLayer(input_size), HiddenLayer(output_size * 3), OutputLayer(output_size))
end

function input!(recognizer::Recognizer, bitmap::Bitmap)
    input!(recognizer.input_layer, bitmap)
    input!(recognizer.hidden_layer, recognizer.input_layer)
    input!(recognizer.output_layer, recognizer.hidden_layer)
end

function update!(recognizer::Recognizer, target_index::Int)::Float64    # reverse propagation
    err = update!(recognizer.output_layer, recognizer.hidden_layer, target_index)
    update!(recognizer.hidden_layer, recognizer.input_layer, recognizer.output_layer)
    return err
end

function train!(nn::NeuralNetwork, bitmaps::Vector{Bitmap})::Float64
    num_bitmaps = length(bitmaps)
    sum_errors = zero(Float64)
    for i = 1:num_bitmaps
        input!(nn, bitmaps[i])
        sum_errors += update!(nn, i)
    end
    return sum_errors / num_bitmaps
end

function classify(nn::NeuralNetwork, bitmap::Bitmap)::Int
    input!(nn, bitmap)
    return argmax(nn.output_layer.outputs)
end


# main

const BITMAP_SIZE = 100         # width * height
const NUM_CHARS = 'Z' - 'A' + 1 # number of alphabet
const TRAINING_ERROR_THRESHOLD = 0.0001
const TRAINING_COUNT_LIMIT = 10000

to_char(index::Int)::Char = index + 'A' - 1

println("[[ file loading ]]")
bitmaps = Vector{Bitmap}([ch |> string |> Bitmap for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"])
recognizer = Recognizer(BITMAP_SIZE, NUM_CHARS)
println("")

println("[[ pattern training ]]")
for cnt = 1:TRAINING_COUNT_LIMIT
    print("training pattern $cnt...")
    err = train!(recognizer, bitmaps)
    println("done. (error: $err)")
    if err < TRAINING_ERROR_THRESHOLD
        break
    end
end
println("")

println("[[ classification testing ]]")
for i = 1:NUM_CHARS
    ch = to_char(i)
    print("test character '$ch'...")
    classified = classify(recognizer, bitmaps[i]) |> to_char
    if classified == ch
        println("correct!")
    else
        println("incorrect! (classified $classified)")
    end
end
println("")
