#Semantic segmentation using a fully invertible hyperbolic neural network.
#This script assumes the labels (segmentation) is available around a few boreholes.
#Training randomly selects a large sub-cubes out of the full data volume.

#For educational purposes, this script uses minimal external tools, and contains
#essentially everything to train a network in a single script;

using Flux
import Flux.Optimise.update!
import Flux: gradient
import Flux.Losses.crossentropy
using InvertibleNetworks #(https://github.com/slimgroup/InvertibleNetworks.jl)
using PyPlot
using LinearAlgebra, Statistics, Random
using NPZ, JLD
using CUDA

use_gpu = true
Random.seed!(4)

#data location
#load seismic data cube (1.5 GB, (1199, 1200, 876))
#download data: https://github.com/olivesgatech/facies_classification_benchmark
#               https://zenodo.org/records/3755060
data_dir  = ""

data_cube_train = npzread(joinpath(data_dir,"train_seismic.npy"));
data_cube_train = convert(Array{Float32,3},data_cube_train)
n = size(data_cube_train);

#load labels
labels_cube_train = zeros(Float32,n...,2)
labels_cube_train[:,:,:,1] = convert(Array{Float32,3},npzread(joinpath(data_dir,"train_labels.npy"))); #put the labels as the first channel

#data normalization (data is already centered around 0 (almost))
data_cube_train .= data_cube_train .- mean(data_cube_train)
data_cube_train .= data_cube_train ./ std(data_cube_train) #std normalization
#data_cube_train .= data_cube_train ./ maximum(abs.(data_cube_train)) #0-1 normalization

#plot some data & labels to see if everyting loads correctly (3rd dimension is depth)
figure(figsize=(10,10));
imshow(data_cube_train[:,100,:,],cmap="Greys",vmin=-3,vmax=3);
imshow(labels_cube_train[:,100,:,1],cmap="jet",alpha=0.2);;colorbar()
savefig("data_label_example.png")

#merge classes (optional)
#If we are just interested in salt - no salt
no_salt_indices = findall(labels_cube_train .!=2)
salt_indices    = findall(labels_cube_train .==2)
labels_cube_train[no_salt_indices] .= 0
labels_cube_train[salt_indices]    .= 1
labels_cube_train[:,:,:,2]         .= 1.0 .- labels_cube_train[:,:,:,1];

#plot some data & labels to see if everyting merged correctly
figure(figsize=(10,10));
imshow(data_cube_train[:,100,:,],cmap="Greys",vmin=-3,vmax=3);
imshow(labels_cube_train[:,100,:,1],cmap="jet",alpha=0.2);;colorbar()
savefig("data_label_example_after_merge.png")

#make sure the full data cube is divisable by 2 sufficiently many times for the Haar transform
data_cube_train   = data_cube_train[:,:,1:248]
labels_cube_train = labels_cube_train[:,:,1:248,:]

#define training and testing labels
  #assume 6 wells for training, with some area around it interpreted
  #assume 3 wells for validation, with some area around it interpreted
n = size(labels_cube_train)
a = round.(Int,shuffle(range(25,stop=375,length=12)))
b = round.(Int,shuffle(range(25,stop=675,length=12)))
train_well_inds_x = a[1:8]
train_well_inds_y = b[1:8]
val_well_inds_x   = a[9:end]
val_well_inds_y   = b[9:end]

area_around_well = 40#pixels-> well location -40 : well location + 40

#generate masks for the entire data volume
label_mask_train = zeros(Float32,size(labels_cube_train)[1:3])
for i=1:length(train_well_inds_x)
  label_mask_train[max(1,train_well_inds_x[i]-area_around_well):min(train_well_inds_x[i]+area_around_well,n[1]),max(1,train_well_inds_y[i]-area_around_well):min(train_well_inds_y[i]+area_around_well,n[2]),:] .= 1f0
end
label_mask_val = zeros(Float32,size(labels_cube_train)[1:3])
for i=1:length(val_well_inds_x)
  label_mask_val[max(1,val_well_inds_x[i]-area_around_well):min(val_well_inds_x[i]+area_around_well,n[1]),max(1,val_well_inds_y[i]-area_around_well):min(val_well_inds_y[i]+area_around_well,n[2]),:] .= 1f0
end

#plot plan view of training and validation labels
figure(figsize=(5,5))
imshow(label_mask_train[:,:,1])
xlabel("X");ylabel("Y");title("Training label locations - planview")
savefig("labels_locs_planview_train.png",bbox_inches="tight")
figure(figsize=(5,5))
imshow(label_mask_val[:,:,1])
xlabel("X");ylabel("Y");title("Validation label locations - planview")
savefig("labels_locs_planview_val.png",bbox_inches="tight")

#make 5 dimensional
data_cube_train   = reshape(data_cube_train,size(data_cube_train)...,1,1)
labels_cube_train = reshape(labels_cube_train,size(labels_cube_train)...,1)

#increase the input-output channel count
data_cube_train = repeat(data_cube_train,outer=[1,1,1,4,1])

#function (for training) that grabs a sub-cube from the full data volume, randomly located in the
#full data volume, but it contains the entire depth range. Returns the sub-cube for
#data, labels and the mask
function GetSubCube(data_cube,label_cube,mask,n_sub)
  n = size(data_cube)
  n_1_min = 1+ceil(Int,n_sub[1]/2); n_1_max = n[1]-ceil(Int,n_sub[1]/2)-1;
  n_2_min = 1+ceil(Int,n_sub[2]/2); n_2_max = n[2]-ceil(Int,n_sub[2]/2)-1;
  #use all of the third dimension
  center_1 = shuffle(n_1_min:n_1_max)[1]
  center_2 = shuffle(n_2_min:n_2_max)[1]
  sub_data_cube  = data_cube[1+center_1-ceil(Int,n_sub[1]/2):center_1+ceil(Int,n_sub[1]/2),1+center_2-ceil(Int,n_sub[2]/2):center_2+ceil(Int,n_sub[2]/2),1:n_sub[3],:,:]
  sub_label_cube = label_cube[1+center_1-ceil(Int,n_sub[1]/2):center_1+ceil(Int,n_sub[1]/2),1+center_2-ceil(Int,n_sub[2]/2):center_2+ceil(Int,n_sub[2]/2),1:n_sub[3],:,:]
  sub_mask       = mask[1+center_1-ceil(Int,n_sub[1]/2):center_1+ceil(Int,n_sub[1]/2),1+center_2-ceil(Int,n_sub[2]/2):center_2+ceil(Int,n_sub[2]/2),1:n_sub[3]]
  sub_data_cube  = sub_data_cube|>gpu
  sub_label_cube = sub_label_cube|>gpu
  sub_mask       = sub_mask|>gpu
  return sub_data_cube, sub_label_cube, sub_mask
end

class_values    = unique(labels_cube_train)
n_class         = length(class_values)
active_channels = 1:n_class

#size of the sub-cube for training
n_sub=[248, 248, 248];
#test if it works well
train_data_test, ~, ~ = GetSubCube(data_cube_train,labels_cube_train,label_mask_train,n_sub)

# change labels to one-hot encoding (explicit coding, could use buildin flux functionality)
n = size(labels_cube_train)
labels_cube_train_1hot = zeros(Int,n[1],n[2],n[3],n_class,1)
for j=1:n_class
  class_indices = findall(labels_cube_train[:,:,:,1,1] .== class_values[j])
  class_cube    = zeros(Int,n[1:3])
  class_cube[class_indices] .= 1
  labels_cube_train_1hot[:,:,:,j,1] .= class_cube
end

#set up neural network
#(0, 8)   -> 0 means no resolution change, maximum block-rank of 8
#(-1, 16) -> -1 means to increase resolution (2x in each dimension) and max of 16 block-rank
#(1, 32)  -> +1 means to decrease resolution (2x in each dimension) and max of 32 block-rank
architecture = ((0, 8), (0, 8), (-1, 16),(0, 16), (0, 16), (-1, 32), (0, 32),(0, 32),(-1, 32),(0, 32),(0, 32),(0, 32),(0, 32),(0, 32),(0, 32),(0, 32),(0, 32),(0, 32),(1, 32),(0, 32),(0, 32),(1, 16),(0, 16),(0, 16),(1, 8),(0, 8),(0, 8),(0, 8),(0, 8),(0, 8))

k = 3   # kernel size
s = 1   # stride
p = 1   # padding
n_chan_in = size(data_cube_train,4)
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
input_tensor_nr_elements         = prod(size(train_data_test))
input_tensor_Gb                  = input_tensor_nr_elements*32/(1000^3)
fully_hyperbolic_inv_network_mem = 3*input_tensor_Gb
non_inv_network_mem              = length(architecture)*fully_hyperbolic_inv_network_mem
println(string("memory for states fully invertible hyperbolic network: ",fully_hyperbolic_inv_network_mem," Gb"))
println(string("memory for states non-invertible network: ",non_inv_network_mem," Gb"))

# define loss functions for the labels and corresponding gradients
#From Flux documentation: "This is mathematically equivalent to crossentropy(softmax(ŷ), y), but is more numerically stable than using functions crossentropy and softmax separately"
#test = (x,y)-> crossentropy(softmax(x,dims=2), y; dims=2, agg=mean,ϵ=eps(x[1]))
CESM   = (x,y)-> Flux.logitcrossentropy(x, y; dims=2, agg=mean)
wCESM  = (x,y,c)-> Flux.logitcrossentropy(x,y;dims=2, agg=x->sum(c .* x))
gCESM  = (x,y)-> gradient(CESM,x,y)
gwCESM = (x,y,c)-> gradient(wCESM,x,y,c)

#define loss function, including basic data augmentation
function loss(HN,use_gpu,X0,label,lossf,lossg,active_channels,mask,flip_dims,permute_dims)

    #random flips (except in depth direction)
    for i in flip_dims
      if rand() > 0.5
        X0          = reverse(X0,dims=i)
        label       = reverse(label,dims=i)
        if isempty(mask) == false
          mask = reverse(mask,dims=i)
        end
      end
    end

    #random data permutation
    if isempty(permute_dims)==false
        permute_dims_augment       = shuffle(permute_dims)
        permutation                = range(1,stop=5,step=1); permutation = convert(Vector{Any},permutation)
        permutation[permute_dims] .=  permute_dims_augment

        X0          = permutedims(X0,permutation)
        if isempty(label) == false
          label         = permutedims(label,permutation)
        end
        if isempty(mask) == false
          mask = permutedims(mask,permutation[1:3])
        end
    end

    #forward propagation
    Y_curr, Y_new, lgdet = HN.forward(X0,X0)

    Y_new = Y_new|>cpu
    label = label|>cpu
    mask  = mask|>cpu
    n_in  = size(Y_new[:,:,:,active_channels,1]) #grab only channels that corresponds to labels, to compute the misfit

    #adapt values in mask from 1&0 to values that take class-imbalance into account (temporary hardcoded for 2 classes)
    nr_pix_class_1 = sum(label[:,:,:,1,1].*mask)
    nr_pix_class_2 = sum(label[:,:,:,2,1].*mask)
    ratio = nr_pix_class_1/nr_pix_class_2
    if ratio>1.0
      class_1_inds        = findall(label[:,:,:,1,1].==1)
      mask[class_1_inds] .= mask[class_1_inds].*(1.0./ratio)
    elseif ratio<1.0
        class_2_inds        = findall(label[:,:,:,2,1].==1)
        mask[class_2_inds] .= mask[class_2_inds].*ratio
    end
    lval         = lossf(reshape(Y_new[:,:,:,active_channels,1],n_in[1]*n_in[2]*n_in[3],n_in[4]),reshape(label[:,:,:,:,1],n_in[1]*n_in[2]*n_in[3],n_in[4]),reshape(mask,n_in[1]*n_in[2]*n_in[3]))
    (grad,dummy) = lossg(reshape(Y_new[:,:,:,active_channels,1],n_in[1]*n_in[2]*n_in[3],n_in[4]),reshape(label[:,:,:,:,1],n_in[1]*n_in[2]*n_in[3],n_in[4]),reshape(mask,n_in[1]*n_in[2]*n_in[3]))
    grad = reshape(grad,n_in[1],n_in[2],n_in[3],n_in[4])
    println("norm grad=", norm(grad,1)/sum(mask))
    Y_new = Y_new|>gpu
    grad  = grad |>gpu

   ΔY_curr= CUDA.zeros(size(Y_new))
   ΔY_curr[:,:,:,active_channels,1] .= grad #the computed gradient is the Lagrangian multiplier for the active channels only, other channels have zero gradient
   ΔY_new = CUDA.zeros(size(Y_new))

   ΔY_curr, ΔY_new, Y_curr, Y_new = HN.backward(ΔY_curr, ΔY_new, Y_curr, Y_new)#reverse propagation and gradient for network weights (saved in HN internally)

   return lval
end

#simple training loop
function Train(HN,batchsize,use_gpu,train_data,val_data,train_labels,val_labels,lossf,lossg,active_channels,mask_train,mask_val,flip_dims,permute_dims,maxiter,opt)
  fval_train     = []
  fval_val       = zeros(maxiter)
  print_counter = 1

  for j=1:maxiter
    for k=1:batchsize
      #grab sub-cube (data, labels & mask)
      train_data_sub_cube, train_labels_sub_cube, mask_train_sub_cube = GetSubCube(train_data,train_labels,mask_train,n_sub)
      #make sure at least some label annotation is included in the randomly selected cube
      while maximum(mask_train_sub_cube)==0
        train_data_sub_cube, train_labels_sub_cube, mask_train_sub_cube = GetSubCube(train_data,train_labels,mask_train,n_sub)
      end
      # Evaluate objective and gradients
      fval = loss(HN,use_gpu,train_data_sub_cube,train_labels_sub_cube,lossf,lossg,active_channels,mask_train_sub_cube,flip_dims,permute_dims)
      push!(fval_train,fval./sum(mask_train_sub_cube))
      println("train f val=", fval./sum(mask_train_sub_cube))
    end

    for p in get_params(HN) #loop over network parameters
        update!(opt, p.data, p.grad) #update network weights
        #update!(lr_decay_fn, p.data, p.grad)
    end
    clear_grad!(HN)

    if mod(j, 20) == 0 #validate every 20 iterations
      #validation data/labels
      fvalepoch_val = 0.0
      for i=1:5#take 5 random subcubes for validation
        val_data_sub_cube, val_labels_sub_cube, mask_val_sub_cube = GetSubCube(val_data,val_labels,mask_val,n_sub)
        while maximum(mask_val_sub_cube)==0
          val_data_sub_cube, val_labels_sub_cube, mask_val_sub_cube = GetSubCube(val_data,val_labels,mask_val,n_sub)
        end
        f = loss(HN,use_gpu,val_data_sub_cube,val_labels_sub_cube,lossf,lossg,active_channels,mask_val_sub_cube,[],[])
        fvalepoch_val = fvalepoch_val + f/sum(mask_val_sub_cube)
      end
      fval_val[print_counter]   = fvalepoch_val/5
       print("Iteration: ", j, "; ftrain = ", fval_train[print_counter], "; fval = ", fval_val[print_counter],"\n")
       print_counter = print_counter + 1
      clear_grad!(HN)
    end

  end

  fval_train   = fval_train[1:print_counter-1]
  fval_val     = fval_val[1:print_counter-1]

  return fval_train, fval_val
end

#re-initialize the network using standard initialization
#(in case some tests above accumulated a gradient or changed the intial weights)
HN = H = NetworkHyperbolic3D(n_chan_in,architecture; α)
if use_gpu==true
  HN           = H |> gpu
end

maxiter      = 120
opt          = Flux.ADAM(1f-3)
flip_dims    = [1,2]
permute_dims = [1,2]
batchsize    = 4

#train network
#1st round
fval_train1, fval_val1 = Train(HN,batchsize,use_gpu,data_cube_train,data_cube_train,labels_cube_train_1hot,labels_cube_train_1hot,wCESM,gwCESM,active_channels,label_mask_train,label_mask_val,flip_dims,permute_dims,maxiter,opt)

#reduce initial stepsize
opt     = Flux.ADAM(1f-4)
fval_train2, fval_val2 = Train(HN,batchsize,use_gpu,data_cube_train,data_cube_train,labels_cube_train_1hot,labels_cube_train_1hot,wCESM,gwCESM,active_channels,label_mask_train,label_mask_val,flip_dims,permute_dims,maxiter,opt)

#turn off data augmentation
flip_dims    = []
permute_dims = []
fval_train3, fval_val3 = Train(HN,batchsize,use_gpu,data_cube_train,data_cube_train,labels_cube_train_1hot,labels_cube_train_1hot,wCESM,gwCESM,active_channels,label_mask_train,label_mask_val,flip_dims,permute_dims,maxiter,opt)
fval_train3, fval_val3 = Train(HN,batchsize,use_gpu,data_cube_train,data_cube_train,labels_cube_train_1hot,labels_cube_train_1hot,wCESM,gwCESM,active_channels,label_mask_train,label_mask_val,flip_dims,permute_dims,maxiter,opt)

figure();
plot([fval_train1;fval_train2;fval_train3])
plot([fval_val1;fval_val2;fval_val3])

#plot training results

  #predict the full data volume, in pieces with overlap
  full_output = zeros(Float32,size(data_cube_train)[1:3]...,2)
  contribution_counter = zeros(Int64,size(data_cube_train)[1:3])
  for i=1:7
    for j=1:19
      println(i)
      println(j)
      data_cube_temp = data_cube_train[25*(i-1)+1:25*(i-1)+248,25*(j-1)+1:25*(j-1)+248,:,:] |> gpu
      data_cube_temp = reshape(data_cube_temp,size(data_cube_temp)...,1)
      contribution_counter[25*(i-1)+1:25*(i-1)+248,25*(j-1)+1:25*(j-1)+248,:,:] .+= 1
      p1, prediction, ~ = HN.forward(data_cube_temp, data_cube_temp)
      prediction = prediction |> cpu
      full_output[25*(i-1)+1:25*(i-1)+248,25*(j-1)+1:25*(j-1)+248,:,:] .+= prediction[:,:,:,1:2]
    end
  end

  full_output[:,:,:,1] ./= contribution_counter
  full_output[:,:,:,2] ./= contribution_counter

  prediction = softmax(full_output[:,:,:,active_channels,1],dims=4);

  #prediction = prediction |> cpu
  prediction_threshold_inds = argmax(prediction[:,:,:,:,1],dims=4)
  prediction_threshold      = zeros(Float32,size(prediction))
  prediction_threshold[prediction_threshold_inds] .= 1

  #Plot 3 cross sections, with training validation labels overlaid
  plot_ind_x = 160
  plot_ind_y = 300
  plot_ind_z = 80

  figure(figsize=(10,10));
  subplot(3,2,1);
  imshow(data_cube_train[:,plot_ind_y,:,1,1]',cmap="Greys",vmin=-3,vmax=3);imshow(labels_cube_train[:,plot_ind_y,:,2,1]',cmap="jet",alpha=0.3);imshow(3.0*label_mask_train[:,plot_ind_y,:]',cmap="Greys",alpha=0.2)
  ;title(string("Data + Labels, Y=",string(plot_ind_y)));ylabel("Z");xlabel("X")
  subplot(3,2,2);
  imshow(data_cube_train[:,plot_ind_y,:,1,1]',cmap="Greys",vmin=-3,vmax=3);imshow(prediction_threshold[:,plot_ind_y,:,1,1]',cmap="jet",alpha=0.3);imshow(3.0*label_mask_train[:,plot_ind_y,:]',cmap="Greys",alpha=0.2)
  ;title(string("Data + Prediction, Y=",string(plot_ind_y)));ylabel("Z");xlabel("X")

  subplot(3,2,3);
  imshow(data_cube_train[plot_ind_x,:,:,1,1]',cmap="Greys",vmin=-3,vmax=3);imshow(labels_cube_train[plot_ind_x,:,:,2,1]',cmap="jet",alpha=0.3);imshow(3.0*label_mask_train[plot_ind_x,:,:]',cmap="Greys",alpha=0.2)
  ;title(string("Data + Labels, X=",string(plot_ind_x)));ylabel("Z");xlabel("Y")
  subplot(3,2,4);
  imshow(data_cube_train[plot_ind_x,:,:,1,1]',cmap="Greys",vmin=-3,vmax=3);imshow(prediction_threshold[plot_ind_x,:,:,1,1]',cmap="jet",alpha=0.3);imshow(3.0*label_mask_train[plot_ind_x,:,:]',cmap="Greys",alpha=0.2)
  ;title(string("Data + Prediction, X=",string(plot_ind_x)));ylabel("Z");xlabel("Y")

  subplot(3,2,5);
  imshow(data_cube_train[:,:,plot_ind_z,1,1],cmap="Greys",vmin=-3,vmax=3);imshow(labels_cube_train[:,:,plot_ind_z,2,1],cmap="jet",alpha=0.3);imshow(3.0*label_mask_train[:,:,plot_ind_z],cmap="Greys",alpha=0.2)
  ;title(string("Data + Labels, Z=",string(plot_ind_z)));ylabel("X");xlabel("Y")
  subplot(3,2,6);
  imshow(data_cube_train[:,:,plot_ind_z,1,1],cmap="Greys",vmin=-3,vmax=3);imshow(prediction_threshold[:,:,plot_ind_z,1,1],cmap="jet",alpha=0.3);imshow(3.0*label_mask_train[:,:,plot_ind_z],cmap="Greys",alpha=0.2)
  ;title(string("Data + Prediction, Z=",string(plot_ind_z)));ylabel("X");xlabel("Y")
  savefig("data_label_prediction_train.png")

  figure(figsize=(10,10));
  subplot(3,2,1);
  imshow(data_cube_train[:,plot_ind_y,:,1,1]',cmap="Greys",vmin=-3,vmax=3);imshow(labels_cube_train[:,plot_ind_y,:,2,1]',cmap="jet",alpha=0.3);imshow(3.0*label_mask_train[:,plot_ind_y,:]',cmap="Greys",alpha=0.2)
  ;title(string("Data + Labels, Y=",string(plot_ind_y)));ylabel("Z");xlabel("X")
  subplot(3,2,2);
  imshow(data_cube_train[:,plot_ind_y,:,1,1]',cmap="Greys",vmin=-3,vmax=3);imshow(prediction[:,plot_ind_y,:,1,1]',cmap="jet",alpha=0.3,vmin=0.4,vmax=0.6);imshow(3.0*label_mask_train[:,plot_ind_y,:]',cmap="Greys",alpha=0.2)
  ;title(string("Data + Prediction, Y=",string(plot_ind_y)));ylabel("Z");xlabel("X")

  subplot(3,2,3);
  imshow(data_cube_train[plot_ind_x,:,:,1,1]',cmap="Greys",vmin=-3,vmax=3);imshow(labels_cube_train[plot_ind_x,:,:,2,1]',cmap="jet",alpha=0.3);imshow(3.0*label_mask_train[plot_ind_x,:,:]',cmap="Greys",alpha=0.2)
  ;title(string("Data + Labels, X=",string(plot_ind_x)));ylabel("Z");xlabel("Y")
  subplot(3,2,4);
  imshow(data_cube_train[plot_ind_x,:,:,1,1]',cmap="Greys",vmin=-3,vmax=3);imshow(prediction[plot_ind_x,:,:,1,1]',cmap="jet",alpha=0.3);imshow(3.0*label_mask_train[plot_ind_x,:,:]',cmap="Greys",alpha=0.2)
  ;title(string("Data + Prediction, X=",string(plot_ind_x)));ylabel("Z");xlabel("Y")

  subplot(3,2,5);
  imshow(data_cube_train[:,:,plot_ind_z,1,1],cmap="Greys",vmin=-3,vmax=3);imshow(labels_cube_train[:,:,plot_ind_z,2,1],cmap="jet",alpha=0.3);imshow(3.0*label_mask_train[:,:,plot_ind_z],cmap="Greys",alpha=0.2)
  ;title(string("Data + Labels, Z=",string(plot_ind_z)));ylabel("Y");xlabel("X")
  subplot(3,2,6);
  imshow(data_cube_train[:,:,plot_ind_z,1,1],cmap="Greys",vmin=-3,vmax=3);imshow(prediction[:,:,plot_ind_z,1,1],cmap="jet",alpha=0.3);imshow(3.0*label_mask_train[:,:,plot_ind_z],cmap="Greys",alpha=0.2)
  ;title(string("Data + Prediction, Z=",string(plot_ind_z)));ylabel("Y");xlabel("X")
  savefig("data_label_probs_train.png")

  #3D plot of seismic data, labels
    using GLMakie

    fig = GLMakie.Figure()
    ax = LScene(fig[1, 1], show_axis=false)

    x = LinRange(0, 401, 401)
    y = LinRange(0, 701, 701)
    z = LinRange(0, 248, 248)

    sgrid = SliderGrid(
        fig[2, 1],
        (label = "yz plane - x axis", range = 1:length(x)),
        (label = "xz plane - y axis", range = 1:length(y)),
        (label = "xy plane - z axis", range = 1:length(z)),
    )

    lo = sgrid.layout
    nc = ncols(lo)

    vol = data_cube_train[:,:,:,1,1]
    vol = reverse(vol,dims=3)
    plt = volumeslices!(ax, x, y, z, vol, colormap=:grays)

    # connect sliders to `volumeslices` update methods
    sl_yz, sl_xz, sl_xy = sgrid.sliders

    on(sl_yz.value) do v; plt[:update_yz][](v) end
    on(sl_xz.value) do v; plt[:update_xz][](v) end
    on(sl_xy.value) do v; plt[:update_xy][](v) end

    set_close_to!(sl_yz, .5length(x))
    set_close_to!(sl_xz, .5length(y))
    set_close_to!(sl_xy, .5length(z))

    # add toggles to show/hide heatmaps
    hmaps = [plt[Symbol(:heatmap_, s)][] for s ∈ (:yz, :xz, :xy)]
    toggles = [Toggle(lo[i, nc + 1], active = true) for i ∈ 1:length(hmaps)]

    map(zip(hmaps, toggles)) do (h, t)
        connect!(h.visible, t.active)
    end

    # cam3d!(ax.scene, projectiontype=Makie.Orthographic)

    fig
    Makie.save("Seismic_data_3D.png", fig)

    fig = GLMakie.Figure()
    ax = LScene(fig[1, 1], show_axis=false)

    x = LinRange(0, 401, 401)
    y = LinRange(0, 701, 701)
    z = LinRange(0, 248, 248)

    sgrid = SliderGrid(
        fig[2, 1],
        (label = "yz plane - x axis", range = 1:length(x)),
        (label = "xz plane - y axis", range = 1:length(y)),
        (label = "xy plane - z axis", range = 1:length(z)),
    )

    lo = sgrid.layout
    nc = ncols(lo)

    vol = labels_cube_train[:,:,:,1,1]
    vol = reverse(vol,dims=3)
    plt = volumeslices!(ax, x, y, z, vol, colormap=:grays)

    # connect sliders to `volumeslices` update methods
    sl_yz, sl_xz, sl_xy = sgrid.sliders

    on(sl_yz.value) do v; plt[:update_yz][](v) end
    on(sl_xz.value) do v; plt[:update_xz][](v) end
    on(sl_xy.value) do v; plt[:update_xy][](v) end

    set_close_to!(sl_yz, .5length(x))
    set_close_to!(sl_xz, .5length(y))
    set_close_to!(sl_xy, .5length(z))

    # add toggles to show/hide heatmaps
    hmaps = [plt[Symbol(:heatmap_, s)][] for s ∈ (:yz, :xz, :xy)]
    toggles = [Toggle(lo[i, nc + 1], active = true) for i ∈ 1:length(hmaps)]

    map(zip(hmaps, toggles)) do (h, t)
        connect!(h.visible, t.active)
    end

    # cam3d!(ax.scene, projectiontype=Makie.Orthographic)

    fig
    Makie.save("Seismic_labels_3D.png", fig)


 ##save trained network parameters
 HN = HN|>cpu
 counter = 1
 W_Array = Array{Any}(undef,length(HN.HL))
 b_Array = Array{Any}(undef,length(HN.HL))
 for i=1:length(HN.HL)
     W_Array[i] = HN.HL[i].W.data
     b_Array[i] = HN.HL[i].b.data
 end

 save("W_Array.jld","W_Array",W_Array)
 save("b_Array.jld","b_Array",b_Array)

 #or load:
 W_Array = load("W_Array.jld","W_Array")
 b_Array = load("b_Array.jld","b_Array")
 HN = HN|>cpu
 counter = 1
 for i=1:length(HN.HL)
    HN.HL[i].W.data = W_Array[i]
    HN.HL[i].b.data = b_Array[i]
 end
 HN = HN|>gpu
