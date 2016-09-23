"""
This example implements a sequence to sequence model with RNN encoder and
decoder, and the model learns to copy a sequence of words. The data is
organised in a sequence-per-line manner. This example can be extended into
an encoder-decoder machine translation model.
"""

module CopySeq

using Knet,AutoGrad,ArgParse,Compat
function main(args=ARGS)
	s = ArgParseSettings()
	s.description="Learning to copy sequences"
	s.exc_handler=ArgParse.debug_handler
	@add_arg_table s begin
	("--datafiles"; nargs='+'; help="If provided, use first file for training, second for dev, others for test.")
	("--hidden"; arg_type=Int; default=256; help="Sizes of one or more LSTM layers.")
	("--embed"; arg_type=Int; default=256; help="Size of the embedding vector.")
	("--epochs"; arg_type=Int; default=3; help="Number of epochs for training.")
	("--batchsize"; arg_type=Int; default=1; help="Number of sequences to train on in parallel.")
	("--lr"; arg_type=Float64; default=1.0; help="Initial learning rate.")
	("--gclip"; arg_type=Float64; default=1.0; help="Value to clip the gradient norm at.")
	("--winit"; arg_type=Float64; default=0.01; help="Initial weights set to winit*randn().")
	("--seed"; arg_type=Int; default=42; help="Random number seed.")
	("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"); help="array type: Array for cpu, KnetArray for gpu")
	("--fast"; action=:store_true; help="skip loss printing for faster run")
	end
	println(s.description)
	isa(args, AbstractString) && (args=split(args))
	o = parse_args(args, s; as_symbols=true)
	println("opts=",[(k,v) for (k,v) in o]...)
	o[:seed] > 0 && srand(o[:seed])
	o[:atype] = eval(parse(o[:atype]))
	global data = Any[]
	for f in o[:datafiles]
		push!(data, process(f, o[:batchsize], o[:atype]))
	end
	vocab = length(data[1][1][2])
	train!(data, vocab; o=o)
end

function train!(data, vocab; o=nothing)
	params = initparams(vocab, o[:hidden], o[:embed], o[:winit], o[:atype])
	state = initstate(o[:atype], o[:hidden], o[:batchsize])
	#TODO: do dev and test
	loss = 0
	for epoch=1:o[:epochs]
		for sentence in data[1]
			loss += train1(params, state, sentence, vocab; o=o)
		end
		println((:epoch,epoch,:loss,loss))
	end
end

function train1(params, state, sentence, vocab; o=nothing)
	gloss = lossgradient(params, sentence, state, vocab; o=o)
	for i=1:length(params)
		params[i] -= o[:lr] * gloss[i]
	end
	for i = 1:length(state)
		isa(state,Value) && error("State should not be a Value.")
		state[i] = getval(state[i])
	end
	return loss(params, sentence, state, vocab; o=o)
end

function loss(params, sentence, state, vocab; o=nothing) #divide loss by num of words
	loss = 0
	decoding=false
	for word in sentence
		encdec(params, word, state, decoding; o=o)
	end
	state[3]=state[1]
	state[4]=state[2]
	decoding=true
	ypred = zeros(o[:batchsize], vocab)
	ypred[vocab] = 1.0
	ypred = convert(o[:atype], ypred)
	for word in sentence
		ygold = word
		ypred = encdec(params, ypred, state, decoding; o=o)
		total = 0.0; count = 0
		atype = typeof(getval(params[1]))
		ynorm = logp(ypred,2) # ypred .- log(sum(exp(ypred),2))
		total += sum(ygold .* ynorm)
		count += size(ygold,1)
		loss += -total / count
	end
	nwords = length(sentence)
	return loss / nwords
end

lossgradient = grad(loss)

function initparams(vocab, hidden, embedding, winit, atype)
	w = Any[]

	push!(w, winit*randn(hidden+embedding, hidden*4))#W_h_encoder
	push!(w, zeros(1, hidden*4))#b_encoder

	push!(w, winit*randn(hidden+embedding, hidden*4))#W_h_decoder
	push!(w, zeros(1, hidden*4))#b_decoder

	push!(w, winit*randn(vocab, embedding)) # W_enc_emb
	push!(w, winit*randn(vocab, embedding)) #	W_dec_emb

	push!(w, winit*randn(hidden, vocab))#W_generate
	push!(w, zeros(1, vocab))#b_generate
	return map(p->convert(atype,p), w)
end

function initstate(atype, hidden, batchsize)
	state = Array(Any, 4)
	for k = 1:2 # 1 for encoder 2 for decoder
		state[2k-1] = zeros(batchsize, hidden)
		state[2k] = zeros(batchsize, hidden)
	end
	return map(s->convert(atype,s), state)
end

function encdec(params, x, state, decoding; o=nothing)
	if !decoding
		emb = x * params[5]
		h = lstm(params, state, 1, emb)
	else
		emb = x * params[6]
		h = lstm(params, state, 3, emb)
		ypred = h[3] * params[7] .+ params[8]
		return exp(logp(ypred))
	end
end

function lstm(param,state,index,input; o=nothing)
	(hidden,cell) = (state[index],state[index+1])
	(weight,bias) = (param[index],param[index+1])
	gates = hcat(input, hidden) * weight + bias
	hsize = size(hidden, 2)
	forget  = sigm(gates[:,1:hsize])
	ingate  = sigm(gates[:,1+hsize:2hsize])
	outgate = sigm(gates[:,1+2hsize:3hsize])
	change  = tanh(gates[:,1+3hsize:end])
	cell    = cell .* forget + ingate .* change
	hidden  = outgate .* tanh(cell)
	(state[index],state[index+1]) = (hidden,cell)
	return state
end

function process(datafile, batchsize, atype)
	#TODO: optimize memory usage
	dict = readvocab(datafile)
	data, line = Any[], Any[]
	word = nothing
	open(datafile) do f
		for l in eachline(f)
				for w in split(l)
					word = zeros(Float64, batchsize, length(dict))
					word[dict[w]] = 1.0
					push!(line, copy(convert(atype, word)))
				end
				push!(data, copy(line))
		end
	end
	return data
end

function readvocab(file) # TODO: test with cmd e.g. `zcat foo.gz`
    d = Dict{Any,Int}()
    open(file) do f
        for l in eachline(f)
            for w in split(l)
                get!(d, w, 1+length(d))
            end
        end
    end
    return d
end

end #module
