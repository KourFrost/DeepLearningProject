
onnx_filename = 'mnist.onnx'

# ONNX to NNgen
dtypes = {}
(outputs, placeholders, variables,
 constants, operators) = ng.from_onnx(onnx_filename,
                                      value_dtypes=dtypes,
                                      default_placeholder_dtype=act_dtype,
                                      default_variable_dtype=weight_dtype,
                                      default_constant_dtype=weight_dtype,
                                      default_operator_dtype=act_dtype,
                                      default_scale_dtype=scale_dtype,
                                      default_bias_dtype=bias_dtype,
                                      disable_fusion=disable_fusion)


# --------------------
    # (2) Assign quantized weights to the NNgen operators
    # --------------------

    if act_dtype.width > 8:
        act_scale_factor = 128
    else:
        act_scale_factor = int(round(2 ** (act_dtype.width - 1) * 0.5))

    input_scale_factors = {'act': act_scale_factor}
    input_means = {'act': imagenet_mean * act_scale_factor}
    input_stds = {'act': imagenet_std * act_scale_factor}

    ng.quantize(outputs, input_scale_factors, input_means, input_stds)

    # --------------------
    # (3) Assign hardware attributes
    # --------------------

    for op in operators.values():
        if isinstance(op, ng.conv2d):
            op.attribute(par_ich=conv2d_par_ich,
                         par_och=conv2d_par_och,
                         par_col=conv2d_par_col,
                         par_row=conv2d_par_row,
                         concur_och=conv2d_concur_och,
                         stationary=conv2d_stationary)

        if isinstance(op, (ng.avg_pool, ng.max_pool,
                           ng.avg_pool_serial, ng.max_pool_serial)):
            op.attribute(par=pool_par)

        if ng.is_elementwise_operator(op):
            op.attribute(par=elem_par)

    # --------------------
    # (4) Verify the DNN model behavior by executing the NNgen dataflow as a software
    # --------------------

    act = placeholders['act']
    out = outputs['out']

    # verification data
    img = np.array(PIL.Image.open('car.png').convert('RGB')).astype(np.float32)
    img = img.reshape([1] + list(img.shape))

    img = img / 255
    img = (img - imagenet_mean) / imagenet_std

    # execution on pytorch
    model_input = np.broadcast_to(img, act_shape)

    if act.perm is not None:
        model_input = np.transpose(model_input, act.reversed_perm)

    model.eval()
    model_out = model(torch.from_numpy(model_input)).detach().numpy()
    if act.perm is not None and len(model_out.shape) == len(act.shape):
        model_out = np.transpose(model_out, act.perm)
    scaled_model_out = model_out * out.scale_factor

    # software-based verification
    vact = img * act_scale_factor
    vact = np.clip(vact,
                   -1.0 * (2 ** (act.dtype.width - 1) - 1),
                   1.0 * (2 ** (act.dtype.width - 1) - 1))
    vact = np.round(vact).astype(np.int64)
    vact = np.broadcast_to(vact, act_shape)

    # compare outputs of hidden layers
    relu_op = [v for k, v in operators.items()
               if isinstance(v, ng.conv2d) and not isinstance(v, ng.matmul)][0]
    maxpool_op = [v for k, v in operators.items()
                  if isinstance(v, (ng.max_pool, ng.max_pool_serial))][0]
    relu_ops = [v for k, v in operators.items()
                if isinstance(v, ng.relu)]
    layer1_0_op = relu_ops[0]
    layer1_op = relu_ops[1]
    layer2_0_op = relu_ops[2]
    layer2_op = relu_ops[3]
    layer3_0_op = relu_ops[4]
    layer3_op = relu_ops[5]
    layer4_0_op = relu_ops[6]
    layer4_op = relu_ops[7]
    avgpool_op = [v for k, v in operators.items()
                  if isinstance(v, (ng.avg_pool, ng.avg_pool_serial))][0]
    fc_op = [v for k, v in operators.items()
             if isinstance(v, ng.matmul)][0]
    sub_ops = [relu_op, maxpool_op,
               layer1_0_op, layer1_op,
               layer2_0_op, layer2_op,
               layer3_0_op, layer3_op,
               layer4_0_op, layer4_op,
               avgpool_op, fc_op]
    sub_outs = ng.eval(sub_ops, act=vact)
    sub_outs = [sub_out.transpose([0, 3, 1, 2]) for sub_out in sub_outs[:-1]] + sub_outs[-1:]
    sub_scale_factors = [sub_op.scale_factor for sub_op in sub_ops]

    model.eval()
    model_relu_out = nn.Sequential(model.conv1,
                                   model.bn1,
                                   model.relu)(torch.from_numpy(model_input)).detach().numpy()
    model_maxpool_out = nn.Sequential(model.conv1,
                                      model.bn1,
                                      model.relu,
                                      model.maxpool)(torch.from_numpy(model_input)).detach().numpy()
