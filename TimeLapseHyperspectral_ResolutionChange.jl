using PyPlot, MAT, Printf
using LinearAlgebra, Random, Statistics
using NNNCVXF #(https://github.com/PetersBas/NNNCVXF)
using Flux
import Flux: gradient
import Flux.Losses.crossentropy
using InvertibleNetworks #(https://github.com/slimgroup/InvertibleNetworks.jl)
#using SetIntersectionProjection  #(https://github.com/slimgroup/SetIntersectionProjection.jl)
using ImageFiltering

# Download data at:
# https://rslab.ut.ac.ir/documents/81960329/82034892/Hyperspectral_Change_Datasets.zip
# from https://rslab.ut.ac.ir/data ("the USA Dataset")

use_gpu = true

data_dir = ""

file = matopen(joinpath(data_dir,"USA_Change_Dataset.mat"));
data = zeros(Float32,307,241,154,2,1)

data[:,:,:,1,1] = read(file,"T1")
data[:,:,:,2,1] = read(file,"T2")

#we are going to lower the resolution by 4x inside the network, so we need to
#coarsen the labels at this point, and then embed them into the larger tensor
label_map = read(file,"Binary")
label_map = reshape(label_map,size(label_map,1),size(label_map,2),1,1)#needs to be 4D tensor for Flux pool functions

#mean filter + round to get rid of single pixel label 'artifacts'
label_map[:,:,1,1] .= mapwindow(median!, label_map[:,:,1,1], (7, 7))
label_map           = Flux.meanpool(label_map,(2,2));
label_map           = round.(label_map) #round back to 0 and 1

#create tensor in which we put the labels as a slice
labels = zeros(Float32,152,120,72,9,1)
labels[:,:,38,1,1] .= label_map[1:152,:,1,1]
labels[:,:,38,9,1]  = 1f0 .- labels[:,:,38,1,1]

#cut data so we can divide by 2 sufficiently many times
data = data[1:304,1:240,1:152,:,:]

#normalize
for i=1:size(data,4)
    data[:,:,:,i,1] .= data[:,:,:,i,1] .- mean(data[:,:,:,i,1]);
    #data[:,:,:,i,1] .= data[:,:,:,i,1] .- minimum(data[:,:,:,i,1]);
    data[:,:,:,i,1] .= data[:,:,:,i,1] ./ maximum(data[:,:,:,i,1]);
end

data = repeat(data,outer=(1,1,1,8,1))

n                  = size(labels)
active_channels    = [1,9] #the labels are embedded as channels 1 and 9, which correspond to a coarser version of the original.

#change data to a vector of examples
dataL    = Vector{Array{Float32,5}}(undef,1)
dataL[1] = data
data     = deepcopy(dataL)
dataL    = []

labelsL    = Vector{Any}(undef,1)
labelsL[1] = labels
labels     = deepcopy(labelsL)
labelsL    = []

architecture = ((0,16),(0,16),(0,16),(0,16),(0,16),(0,16),(-1,16),(0,16),(0,16),(0,16),(0,16),(0,16),(0,16),(0,16),(0,16),(0,16),(0,16),(0,16))

k = 3   # kernel size
s = 1   # stride
p = 1   # padding
n_chan_in = size(data[1],4)
α = 0.2^2 #artificial time-step in the nonlinear telegraph equation discretization that is the neural network
HN = H = NetworkHyperbolic3D(n_chan_in,architecture; α)

#compute memory for convolutional kernels & what it would be without block-low-rank layers
conv_kernel_mem_BLR = 0
conv_kernel_mem_reg = 0
for i=1:length(H.HL)
  conv_kernel_mem_BLR = conv_kernel_mem_BLR + 27*prod(size(H.HL[i].W)[4:5])*32/(1000^3)
  conv_kernel_mem_reg = conv_kernel_mem_reg + 27*maximum(size(H.HL[i].W)[4:5])^2*32/(1000^3)
end

println(string("memory for conv. kernels - block-low-rank layers: ",conv_kernel_mem_BLR," Gb"))
println(string("memory for conv. kernels - standard layers: ",conv_kernel_mem_reg," Gb"))


#compute memory for states, and what it would be for a non-reversible direct-equivalent
input_tensor_nr_elements         = prod(size(data[1]))
input_tensor_Gb                  = input_tensor_nr_elements*32/(1000^3)
fully_hyperbolic_inv_network_mem = 3*input_tensor_Gb
non_inv_network_mem              = length(architecture)*fully_hyperbolic_inv_network_mem
println(string("memory for states fully invertible hyperbolic network: ",fully_hyperbolic_inv_network_mem," Gb"))
println(string("memory for states non-invertible network: ",non_inv_network_mem," Gb"))

pos_inds        = findall(labels[1][:,:,38,1,1].>0)
pos_inds_select = shuffle(pos_inds)[1:35]
neg_inds        = findall(labels[1][:,:,38,1,1].==0)
neg_inds_select = shuffle(neg_inds)[1:35]

active_z_slice = 38


######### Point annotations (using standard cross-entropy loss) for both classes WITHOUT other prior knowledge ##########

output_samples_train = Vector{Any}(undef,1)
output_samples_train[1] = zeros(Float32,size(labels[1])[1:2])
output_samples_train[1][pos_inds_select[1:25]] .= 1
output_samples_train[1][neg_inds_select[1:25]] .= 1

output_samples_val = Vector{Any}(undef,1)
output_samples_val[1] = zeros(Float32,size(labels[1])[1:2])
output_samples_val[1][pos_inds_select[26:end]] .= 1
output_samples_val[1][neg_inds_select[26:end]] .= 1

#dummy projectors (no additional prior info is enforced)
P    = Vector{Vector{Any}}(undef,1)
P[1] = Vector{Any}(undef,2)
P[1][1] = x -> x
P[1][2] = x -> x

if use_gpu==true
  HN   = H |> gpu  #move network to use_gpu
  data = data|>gpu #move data to gpu
end

logs = Log() #create structure of logs of misfits

#fill unused data/labels with empty arrays
val_data      = deepcopy(data)
train_labels  = deepcopy(labels)#[[]]
val_labels    = deepcopy(labels)

# define loss functions for the labels and corresponding gradients
CESM   = (x,y)-> crossentropy(softmax(x,dims=3), y; dims=3, agg=sum,ϵ=eps(x[1]))
wCESM  = (x,y,c)-> crossentropy(softmax(x,dims=3), y; dims=3, agg=x->sum(c .* x),ϵ=eps(x[1]))
gCESM  = (x,y)-> gradient(CESM,x,y)
gwCESM = (x,y,c)-> gradient(wCESM,x,y,c)

#set up the training options
#set up the training options
TrainOptions = TrOpts()
TrainOptions.eval_every      = 5
TrainOptions.batchsize       = 1
TrainOptions.use_gpu         = use_gpu
TrainOptions.lossf           = []
TrainOptions.lossg           = []
TrainOptions.active_channels = active_channels
TrainOptions.flip_dims       = []
TrainOptions.permute_dims    = []

TrainOptions.lossf   = wCESM
TrainOptions.lossg   = gwCESM
TrainOptions.alpha   = 0f0
TrainOptions.opt     = Flux.Momentum(1f-3,0.9)
TrainOptions.maxiter = 35

#train the network; returns the loss, network parameters are updated inplace so HN is updated after training
logs = Train(HN,logs,TrainOptions,data,val_data,train_labels,val_labels,P,output_samples_train,output_samples_val,active_z_slice)

TrainOptions.opt     = Flux.Momentum(1f-4,0.9)
TrainOptions.maxiter = 30
logs = Train(HN,logs,TrainOptions,data,val_data,train_labels,val_labels,P,output_samples_train,output_samples_val,active_z_slice)


#plot the distance function per iteration
iter_ax = range(0,step=TrainOptions.eval_every,length=length(logs.train))

figure(figsize=(5,4));
semilogy(iter_ax,logs.train,label="train");#title("Training loss labels");
semilogy(iter_ax,logs.val,"--",label="validation");legend();title("Cross-entropy loss labels");
xlabel("Iteration number");
tight_layout()
savefig("loss_hyperspectral labels.png")

#plot data and results
PlotDataLabelPredictionHyperspectral(1,data,labels,HN,TrainOptions.active_channels,active_z_slice,pos_inds_select,neg_inds_select,"train")
