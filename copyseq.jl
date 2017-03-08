# TODO:
# - Solve problem that causes the generator to copy only a few sentences
# - Update gcheck to match current format

for p in ("Knet","AutoGrad","ArgParse","Compat")
    Pkg.installed(p) == nothing && Pkg.add(p)
end
"""
This example implements a sequence to sequence model with RNN encoder and
decoder, and the model learns to copy a sequence of words. The data is
organised in a sequence-per-line manner. This example can be extended into
an encoder-decoder machine translation model.

Example usage:
* `julia copyseq.jl --data foo.txt`: uses foo.txt to train.
* `julia copyseq.jl --data foo.txt bar.txt`: uses foo.txt for training
  and bar.txt for validation.  Any number of files can be specified,
  the first two will be used for training and validation, the rest for
  testing.
* `julia copyseq.jl --best foo.jld --save bar.jld`: saves the best
  model (according to validation set) to foo.jld, last model to
  bar.jld.
* `julia copyseq.jl --load foo.jld --copy bar.txt`: generates a copy of
bar.txt into bar-copy.txt using the model in foo.jld.
* `julia copyseq.jl --help`: describes all available options.
"""
module CopySeq

using Knet,AutoGrad,ArgParse,Compat

include("preprocess.jl")

function main(args=ARGS)
	s = ArgParseSettings()
	s.description="Learning to copy sequences"
	s.exc_handler=ArgParse.debug_handler
	@add_arg_table s begin
	("--datafiles"; nargs='+'; help="If provided, use first file for training, second for dev, others for test.")
	("--loadfile"; help="Initialize model from file")
	("--savefile"; help="Save final model to file")
	("--bestfile"; help="Save best model to file")
	("--dictfile"; help="Dictionary file, first datafile used if not specified")
	("--copy"; help="Generates a copy of the provided file")
	("--hidden"; arg_type=Int; default=256; help="Sizes of one or more LSTM layers.")
	("--embed"; arg_type=Int; default=256; help="Size of the embedding vector.")
	("--epochs"; arg_type=Int; default=3; help="Number of epochs for training.")
	("--batchsize"; arg_type=Int; default=1; help="Number of sequences to train on in parallel.")
	("--decay"; arg_type=Float64; default=1.0; help="Learning rate decay.")
	("--lr"; arg_type=Float64; default=2.0; help="Initial learning rate.")
	("--gclip"; arg_type=Float64; default=5.0; help="Value to clip the gradient norm at.")
	("--winit"; arg_type=Float64; default=0.10; help="Initial weights set to winit*randn().")
	("--gcheck"; arg_type=Int; default=0; help="Check N random gradients.")
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
if any(f->(o[f]!=nothing), (:loadfile, :savefile, :bestfile))
  Pkg.installed("JLD") == nothing && Pkg.add("JLD")
  eval(Expr(:using,:JLD))
end
if o[:loadfile]==nothing
	global data = Any[]
	push!(data, Data(o[:datafiles][1]; batchsize=o[:batchsize], vocabfile=nothing))
	dict=data[1].word_to_index
	vocab=length(dict)
	if length(o[:datafiles])>1
		for i=2:length(o[:datafiles])
			push!(data, Data(o[:datafiles][i]; batchsize=o[:batchsize], word_to_index=dict))
		end
	end
	model = initparams(vocab, o[:hidden], o[:embed], o[:winit], o[:atype])
else
	info("Loading model from $(o[:loadfile])")
	data = load(o[:loadfile], "data")
	model = map(p->convert(o[:atype], p), load(o[:loadfile], "model"))
	dict=data[1].word_to_index
	vocab=length(dict)
end
if !isempty(data)
	train!(data, model, vocab, o)
end
if o[:savefile] != nothing
	info("Saving the last model to $([:savefile])")
	model32 = map(p->convert(Array{Float32}, p), model)
	save(o[:savefile], "model", model32, "data", data)
end
if o[:copy] != nothing
	state = initstate(o[:atype], o[:hidden], 1)
	copydata = Data(o[:copy]; batchsize=1, word_to_index=dict)
	copysequence(model, copy(state), vocab, copydata, o)
end
println(model[5]-model[6])
end

function train!(data, model, vocab, o)
	state = initstate(o[:atype], o[:hidden], o[:batchsize])
	if o[:fast]
		@time (for epoch=1:o[:epochs]
					for batch in data[1]
					    train1(model, batch, copy(state), vocab, o; gscale=o[:lr], gclip=o[:gclip])
					end
				end; gpu()>=0 && Knet.cudaDeviceSynchronize())
		return
	end
#	losses = map(d->test(d, model, copy(state), vocab, o), data)
#	println((:epoch,0,:loss,losses...))
	@time(for epoch=1:o[:epochs]
		lss = 0
		batch_cnt = 0
		converted_data = Any[]
		for batch in data[1]
			converted_data = map(w->(convert(o[:atype]), w), batch)
	    	lss += train1(model, converted_data, copy(state), vocab, o; gscale=o[:lr], gclip=o[:gclip])
			batch_cnt += 1
			if o[:gcheck] > 0 && batch_cnt == 1 #check gradients only for the first batch
				gradcheck(loss, model, batch, batch[2:end], copy(state), vocab, o; gcheck=o[:gcheck])
			end
			empty!(converted_data)
	  	end
		losses=Array(Float32, length(data)-1)
		for i=2:length(data)
			losses[i-1] = test(data[i], model, state, vocab, o)
		end
		println((:epoch,epoch,:trn_loss,lss/batch_cnt, :test_loss, losses...))
	end)
end

function train1(params, batch, state, vocab, o; gscale=1.0, gclip=0)
	gloss = lossgradient(params, batch, batch[2:end], copy(state), vocab, o)
	if gclip > 0
		gnorm = sqrt(mapreduce(sumabs2, +, 0, gloss))
		if gnorm > gclip
			gscale *= gclip / gnorm
		end
	end
	for k in 1:length(params)
	 	axpy!(-gscale, gloss[k], params[k])
	end
	isa(state,Vector{Any}) || error("State should not be Boxed.")
	# The following is needed in case AutoGrad boxes state values during gradient calculation
	for i = 1:length(state)
    	state[i] = AutoGrad.getval(state[i])
  	end
	lss = loss(params, batch, batch[2:end], copy(state), vocab, o)
	return lss
end

function test(data, params, state, vocab, o)

	converted_data = Any[]
	converted_batch = Any[]
	for batch in data
		converted_batch = map(w->convert(o[:atype], w), batch)
		push!(converted_data, converted_batch)
		empty!(converted_batch)
	end

	lss = 0
	batch_cnt = 0
	for batch in converted_data
		lss += loss(params, batch, batch[2:end], copy(state), vocab, o)
		batch_cnt += 1
	end
	return lss/batch_cnt
end

function loss(params, sentence, ysentence, state, vocab, o)
	#encoder
	total = 0.0; count = 0
	decoding=false
	for word in sentence
			state = encdec(params, word, state, decoding)
	end
	# copy encoder's hidden states to decoder
	state[3]=copy(state[1])
	state[4]=copy(state[2])
	decoding=true
  for i=1:length(ysentence)
    #println(get_word(convert(Array{Float32}, word)[1,:]))
		state, ypred = encdec(params, sentence[i], state, decoding) #predict
    try
    # println(ypred)
    # println(get_word(convert(Array{Float32}, ypred)[1,:]))
    catch
    end
		ynorm = logp(ypred,2)
		total += sum(ysentence[i] .* ynorm)
		count += size(ysentence[i],1)
  	end
	return -total/count
end

lossgradient = grad(loss)

function initparams(vocab, hidden, embedding, winit, atype)
	w = Array(Any, 8)
	w[1] = winit*randn(hidden+embedding, hidden*4)#W_h_encoder
	w[2] = zeros(1, hidden*4)#b_encoder

	w[3] = winit*randn(hidden+embedding, hidden*4)#W_h_decoder
	w[4] = zeros(1, hidden*4)#b_decoder

	w[5] = winit*randn(vocab, embedding) # W_enc_emb
	w[6] = winit*randn(vocab, embedding) #	W_dec_emb

	w[7] = winit*randn(hidden, vocab)#W_generate
	w[8] =  zeros(1, vocab)#b_generate
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

function encdec(params, x, state, decoding)
	if !decoding
		emb = x * params[5]
		state = lstm(params, state, 1, emb)
	else
		emb = x * params[6]
		state = lstm(params, state, 3, emb)
		ypred = state[3] * params[7] .+ params[8]
		return state, ypred
	end
end

function lstm(param,state,index,input)
	(hidden,cell) = (state[index],state[index+1])
	(weight,bias) = (param[index],param[index+1])
	gates = hcat(input, hidden) * weight .+ bias
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

function copysequence(params, ostate, vocab, copydata, o)
	converted_batch = Any[]
	for batch in copydata
		state = copy(ostate)
		print("real sentence: ")
		map(w->print(get_word(w)), batch)
		converted_batch = map(w->convert(o[:atype], w), batch)
		println()
	    println("word count:", size(converted_batch))
    	println("word size:", size(converted_batch[1]))
    	o[:batchsize] = 1
		println("loss: ", loss(params, converted_batch, converted_batch[2:end], copy(ostate), vocab, o))
		decoding=false
		for word in converted_batch
			state = encdec(params, word, state, decoding)
		end
		# copy encoder's hidden states to decoder
		state[3]=state[1]
		state[4]=state[2]
		decoding=true
		# give <s> as the first token into decoder
		ypred = zeros(1, vocab)
		ypred[1, 1] = 1
		input = convert(o[:atype], ypred)
    	ind = 0
    	word_cnt=0
	    while (ind!=2 && word_cnt<300) #until end of sequence
    		word_cnt=word_cnt+1
			state, ypred = encdec(params, input, state, decoding) #predict
    		ind = indmax(convert(Array{Float32}, ypred))
    		input = zeros(1, vocab)
  			input[1, ind] = 1
  			input = convert(o[:atype], input)
     		print(copydata.index_to_word[ind], " ")
    		#print(get_word(input), " ")
		end
    	println()
    	empty!(converted_batch)
	end
end

function get_word(v)
    ind = indmax(convert(Array{Float32}, v))
    word = data[1].index_to_word[ind]
    return word
end

!isinteractive() && !isdefined(Core.Main,:load_only) && main(ARGS)

end #module
