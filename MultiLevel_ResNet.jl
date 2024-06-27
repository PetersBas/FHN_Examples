export ML_ResNet2, Initialize_ML_ResNet
import Flux.normalise
using InvertibleNetworks

## multilevel ResNet below
function Initialize_ML_ResNet(n_chan_data,n_resolutions,n_conv_per_resolution,n_conv_lowest_level)
    total_convs = sum(n_conv_per_resolution)+n_conv_lowest_level
    K = Vector{Array{Float32,5}}(undef,total_convs)
    k_ind = 1
    n_chan = deepcopy(n_chan_data)
    n_chan_prev = deepcopy(n_chan)
    scale = 1f-1
    for i=1:n_resolutions
        for j=1:n_conv_per_resolution[i]
            K[k_ind] = Flux.convfilter((3,3,3),n_chan_prev=>n_chan)
            k_ind = k_ind + 1
            n_chan_prev = deepcopy(n_chan)
        end
        n_chan = n_chan * 8
    end

    for j=1:n_conv_lowest_level
        K[k_ind] = Flux.convfilter((3,3,3),n_chan_prev=>n_chan)
        n_chan_prev = deepcopy(n_chan)
        k_ind = k_ind + 1

    end

    for i=1:length(K)
        println(size(K[i]))
    end

    return K 
end 

function ML_ResNet2(x,h,K,n_resolutions,n_conv_per_resolution,n_conv_lowest_level)
    k_ind = 1
    #downward pass
    for i=1:n_resolutions
        for j=1:n_conv_per_resolution[i]
            #println(size(x))
            if size(K[k_ind])[end] == size(K[k_ind])[end-1]
                cdims = InvertibleNetworks.DCDims(x, K[k_ind]; stride=1, padding=1)
                x = x - h*∇conv_data(relu.(conv(x,K[k_ind],pad=1)),K[k_ind],cdims)
            else
                x = relu.(conv(x,K[k_ind],pad=1))
            end
            k_ind = k_ind + 1
            
        end
    
        #move one resolution down
        x = maxpool(x,(2,2,2),stride=2)#max pooling with strides of two
    end

    #resnet layers on lowest level
    for j=1:n_conv_lowest_level
        #println(size(x))
        if size(K[k_ind])[end] == size(K[k_ind])[end-1]
            cdims = InvertibleNetworks.DCDims(x, K[k_ind]; stride=1, padding=1)
            x = x - h*∇conv_data(relu.(conv(x,K[k_ind],pad=1)),K[k_ind],cdims)
        else
            x = relu.(conv(x,K[k_ind],pad=1))
        end
        k_ind = k_ind + 1
    end

    return x
end 
