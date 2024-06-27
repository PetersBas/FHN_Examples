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

Random.seed!(2)

use_gpu = true

data_dir = ""

#Read data
file = matopen(joinpath(data_dir,"USA_Change_Dataset.mat"));
data = zeros(Float32,368, 288, 184,2,1)

temp = read(file,"T1");
temp = reshape(temp,size(temp)...,1,1);
temp = convert(Array{Float32,5},temp);
temp = Flux.upsample_trilinear(temp,(1.2, 1.199, 1.2))
data[:,:,:,1,1] = temp
temp = read(file,"T2");
temp = reshape(temp,size(temp)...,1,1);
temp = convert(Array{Float32,5},temp);
temp = Flux.upsample_trilinear(temp,(1.2, 1.199, 1.2))
data[:,:,:,2,1] = temp

#we are going to lower the resolution by 4x inside the network, so we need to
#coarsen the labels at this point, and then embed them into the larger tensor
label_map = read(file,"Binary")
label_map = reshape(label_map,size(label_map)...,1,1)#needs to be 4D tensor for Flux pool functions

#mean filter + round to get rid of single pixel label 'artifacts'
label_map[:,:,1,1] .= mapwindow(median!, label_map[:,:,1,1], (7, 7))
label_map           = Flux.meanpool(label_map,(2,2));
label_map = Flux.upsample_bilinear(label_map,(1.205, 1.2))
label_map           = round.(label_map) #round back to 0 and 1

#create tensor in which we embed the labels as a slice
labels = zeros(Float32,184, 144,72,9,1)
labels[:,:,38,1,1] .= label_map[:,:,1,1]
labels[:,:,38,9,1]  = 1f0 .- labels[:,:,38,1,1]

#normalize
for i=1:size(data,4)
    data[:,:,:,i,1] .= data[:,:,:,i,1] .- mean(data[:,:,:,i,1]);
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
  conv_kernel_mem_BLR = conv_kernel_mem_BLR + 27*prod(size(H.HL[i].W)[4:5])*4/(1000^3)
  conv_kernel_mem_reg = conv_kernel_mem_reg + 27*maximum(size(H.HL[i].W)[4:5])^2*4/(1000^3)
end
println(string("memory for conv. kernels - block-low-rank layers: ",conv_kernel_mem_BLR," Gb"))
println(string("memory for conv. kernels - standard layers: ",conv_kernel_mem_reg," Gb"))

#compute memory for states, and what it would be for a non-reversible direct-equivalent
input_tensor_nr_elements         = prod(size(data[1]))
input_tensor_GB                  = input_tensor_nr_elements*4/(1000^3)
fully_hyperbolic_inv_network_mem = 3*input_tensor_GB
non_inv_network_mem              = length(architecture)*input_tensor_GB
println(string("memory for states fully invertible hyperbolic network: ",fully_hyperbolic_inv_network_mem," GB"))
println(string("memory for states non-invertible network: ",non_inv_network_mem," GB"))

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
train_labels  = deepcopy(labels)#[[]]
val_labels    = deepcopy(labels)

# define loss functions for the labels and corresponding gradients
CESM   = (x,y)-> crossentropy(softmax(x,dims=3), y; dims=3, agg=sum,ϵ=eps(x[1]))
wCESM  = (x,y,c)-> crossentropy(softmax(x,dims=3), y; dims=3, agg=x->sum(c .* x),ϵ=eps(x[1]))
gCESM  = (x,y)-> gradient(CESM,x,y)
gwCESM = (x,y,c)-> gradient(wCESM,x,y,c)

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

#test fwd
#plt_ind=1
#p1, prediction, ~ = HN.forward(data[plt_ind], data[plt_ind]);

#train the network; returns the loss, network parameters are updated inplace so HN is updated after training
logs = Train(HN,logs,TrainOptions,data,data,train_labels,val_labels,P,output_samples_train,output_samples_val,active_z_slice,P)

TrainOptions.opt     = Flux.Momentum(1f-4,0.9)
TrainOptions.maxiter = 30
logs = Train(HN,logs,TrainOptions,data,data,train_labels,val_labels,P,output_samples_train,output_samples_val,active_z_slice,P)

#plot the loss
iter_ax = range(0,step=TrainOptions.eval_every,length=length(logs.train))

figure(figsize=(3.8,5));
semilogy(iter_ax,logs.train/50,label="train");#title("Training loss labels");
semilogy(iter_ax,logs.val/20,"--",label="validation");legend();title("Cross-entropy loss labels");
xlabel("Iteration number");
savefig("loss_hyperspectral_labels.png",bbox_inches="tight")

#plot data and results
pos_inds_plot = findall(labels[1][:,:,38,active_channels[1],1].*output_samples_train[1].>0)
neg_inds_plot = findall(labels[1][:,:,38,active_channels[2],1].*output_samples_train[1].>0)
PlotDataLabelPredictionHyperspectral(1,data,labels,HN,TrainOptions.active_channels,active_z_slice,pos_inds_plot,neg_inds_plot,"train")

# ####################################################################
# ####################################################################
# ## hypothetically, using the largest possible non-invertible version of a fully hyperbolic convolutional network
# if it were implemented using automatic differentiation on a 24GB NVIDIA GeForce RTX 3090
# plt_ind=1
# HyperFWD = x-> HN.forward(x, x)[2]
# #p1, prediction, ~ = HN.forward(data[plt_ind], data[plt_ind]);
# #out = HyperFWD(data[plt_ind]);
# loss = (x,HN,labels,mask) -> wCESM(HN.forward(x, x)[2][:,:,active_z_slice,active_channels,1],labels,mask)
# #test=Net_FWD(data[1],K);
# test_val_grad = Flux.withgradient(loss, data[1], HN, train_labels[1][:,:,active_z_slice,active_channels,1]|>gpu, output_samples_train[1]|>gpu);

# ####################################################################
# ####################################################################
#  using the largest possible ResNet on a 24GB NVIDIA GeForce RTX 3090

include("MultiLevel_ResNet.jl")
h_resnet                = 0.2f0 |> gpu

#same as hypernet, too large for memory ()
# n_resolutions         = 1
# n_conv_per_resolution = [6]
# n_conv_lowest_level   = 12
# n_chan_data           = 16

#n_conv_per_resolution=[6]; n_conv_lowest_level=1 #out of memory
#n_conv_per_resolution=[5]; n_conv_lowest_level=3 #is also too large for memory
#n_conv_per_resolution=[3]; n_conv_lowest_level=5 #out of memory

#test the following networks: 
n_conv_per_resolution=[3]; n_conv_lowest_level=4 #(network 1)
#n_conv_per_resolution=[2]; n_conv_lowest_level=5 #(network 2)
#n_conv_per_resolution=[5]; n_conv_lowest_level=2 #(network 3)


n_resolutions         = 1
n_chan_data           = 16
n_conv_per_resolution=[3]; n_conv_lowest_level=4

K = Initialize_ML_ResNet(n_chan_data,n_resolutions,n_conv_per_resolution,n_conv_lowest_level);
K = K |>gpu;

#compute memory estimate

#compute memory for convolutional kernels for multi-level ResNet
conv_kernel_mem = 0.0
for i=1:length(K)
  conv_kernel_mem_reg += ( prod(size(K[i])[1:3])*prod(size(K[i])[4:5]) )
end
#conv_kernel_mem_reg #currently nr of elements Float32
conv_kernel_mem_reg *= 4 #4 bytes per Float32
conv_kernel_mem_reg /= 1000^3 #->GB
println(string("memory for conv. kernels - multi-level ResNet: ",conv_kernel_mem_reg," Gb"))

#compute memory for states for a multi-level ResNet
input_tensor_nr_elements = prod(size(data[1]))
state_size_per_chan      = deepcopy(input_tensor_nr_elements)/size(data[1],4) #input_tensor/nchannels
total_state_size         = 0.0
for i=1:length(K)
  if size(K[i])[5] != size(K[i])[4]
    state_size_per_chan = state_size_per_chan/8 #after maxpooling in 3D
  end
  total_state_size += state_size_per_chan*size(K[i])[5]
end
#total_state_size #currently nr of elements Float32
non_inv_network_mem = 4*total_state_size #4 bytes per Float32
non_inv_network_mem /= 1000^3 #->GB)
println(string("memory for states non-invertible network: ",non_inv_network_mem," Gb")) #only for states (not including Lagrangian multipliers and gradients)


#test forward propagation
Net_FWD = (x,Ckernels) -> ML_ResNet2(x,h_resnet,Ckernels,n_resolutions,n_conv_per_resolution,n_conv_lowest_level);
loss = (x,Ckernels,labels,mask) -> wCESM(ML_ResNet2(x,h_resnet,Ckernels,n_resolutions,n_conv_per_resolution,n_conv_lowest_level)[:,:,active_z_slice,active_channels,1],labels,mask)
#test=Net_FWD(data[1],K);
#test_val_grad = Flux.withgradient(loss, data[1], K, train_labels[1][:,:,active_z_slice,active_channels,1]|>gpu, output_samples_train[1]|>gpu);
#Flux.update!(TrainOptions.opt, K[1], test_val_grad[2][2][1]) #test update of model parameters 

#set up the training options - for UNet
TrainOptions = TrOpts()
TrainOptions.eval_every      = 5
TrainOptions.batchsize       = 1
TrainOptions.use_gpu         = use_gpu
TrainOptions.lossf           = []
TrainOptions.lossg           = []
TrainOptions.active_channels = active_channels
TrainOptions.flip_dims       = []
TrainOptions.permute_dims    = []

TrainOptions.alpha   = 0f0
TrainOptions.opt     = Flux.Momentum(1f-3,0.9)
TrainOptions.maxiter = 35

K, ftrain1, f_val1 = Train_AD(K,loss,TrainOptions,data,data,train_labels,val_labels,output_samples_train,output_samples_val,active_z_slice);

TrainOptions.opt     = Flux.Momentum(1f-4,0.9)
TrainOptions.maxiter = 30
K, ftrain1_b, f_val1_b = Train_AD(K,loss,TrainOptions,data,data,train_labels,val_labels,output_samples_train,output_samples_val,active_z_slice);

#another multilevel resnet 
n_conv_per_resolution=[2]; n_conv_lowest_level=5 #(2)
K = Initialize_ML_ResNet(n_chan_data,n_resolutions,n_conv_per_resolution,n_conv_lowest_level);
K = K |>gpu;

#set up the training options - for multilevel resnet
TrainOptions = TrOpts()
TrainOptions.eval_every      = 5
TrainOptions.batchsize       = 1
TrainOptions.use_gpu         = use_gpu
TrainOptions.lossf           = []
TrainOptions.lossg           = []
TrainOptions.active_channels = active_channels
TrainOptions.flip_dims       = []
TrainOptions.permute_dims    = []

TrainOptions.alpha   = 0f0
TrainOptions.opt     = Flux.Momentum(1f-3,0.9)
TrainOptions.maxiter = 35

K, ftrain2, f_val2 = Train_AD(K,loss,TrainOptions,data,data,train_labels,val_labels,output_samples_train,output_samples_val,active_z_slice);

TrainOptions.opt     = Flux.Momentum(1f-4,0.9)
TrainOptions.maxiter = 30
K, ftrain2_b, f_val2_b = Train_AD(K,loss,TrainOptions,data,data,train_labels,val_labels,output_samples_train,output_samples_val,active_z_slice);

#another multilevel resnet 
n_conv_per_resolution=[5]; n_conv_lowest_level=2 #(3)
K = Initialize_ML_ResNet(n_chan_data,n_resolutions,n_conv_per_resolution,n_conv_lowest_level);
K = K |>gpu;

#set up the training options - for multilevel resnet
TrainOptions = TrOpts()
TrainOptions.eval_every      = 5
TrainOptions.batchsize       = 1
TrainOptions.use_gpu         = use_gpu
TrainOptions.lossf           = []
TrainOptions.lossg           = []
TrainOptions.active_channels = active_channels
TrainOptions.flip_dims       = []
TrainOptions.permute_dims    = []

TrainOptions.alpha   = 0f0
TrainOptions.opt     = Flux.Momentum(1f-3,0.9)
TrainOptions.maxiter = 35

K, ftrain3, f_val3 = Train_AD(K,loss,TrainOptions,data,data,train_labels,val_labels,output_samples_train,output_samples_val,active_z_slice);

TrainOptions.opt     = Flux.Momentum(1f-4,0.9)
TrainOptions.maxiter = 30
K, ftrain3_b, f_val3_b = Train_AD(K,loss,TrainOptions,data,data,train_labels,val_labels,output_samples_train,output_samples_val,active_z_slice);

#
ax1 = [1 ; 5:5:35 ; 36 ; 40:5:65]
#ax2 = 
figure(figsize=(5,3));
#semilogy(ax1,logs.train/50,label="train - hyper",c="b");#title("Training loss labels");
semilogy(ax1,logs.val/20,"--",label="validation - HyperNet",c="b");legend();title("Cross-entropy loss");
#semilogy(ax1,[ftrain1 ;ftrain1_b][ax1]/50,label="train - ResNet 1",c="r");#title("Training loss labels");
semilogy(ax1,[f_val1[[1 ; 5:5:35]] ;f_val1_b[[1 ; 5:5:30]]]/20,"--",label="validation - ResNet 1",c="r")
#semilogy(ax1,[ftrain2 ;ftrain2_b][ax1]/50,label="train - ResNet 2",c="g");#title("Training loss labels");
semilogy(ax1,[f_val2[[1 ; 5:5:35]] ;f_val2_b[[1 ; 5:5:30]]]/20,"--",label="validation - ResNet 2",c="g")
#semilogy(ax1,[ftrain3 ;ftrain3_b][ax1]/50,label="train - ResNet 3",c="k");#title("Training loss labels");
semilogy(ax1,[f_val3[[1 ; 5:5:35]] ;f_val3_b[[1 ; 5:5:30]]]/20,"--",label="validation - ResNet 3",c="k")
;legend();
xlabel("Iteration number");
#tight_layout()
savefig("loss_hyperspectral_labels_compare.png",bbox_inches="tight")
