# coding: utf-8

'''
@Author: ywsong
@Date: 20210-5-10
'''

def check_model(model_file):
    import onnx

    assert isinstance(model_file, str)
    
    onnx_model = onnx.load(model_file) 

    print('check_model', model_file)
    onnx.checker.check_model(onnx_model)
    

def print_info(model_file):
    import onnxruntime

    assert isinstance(model_file, str) 

    sess = onnxruntime.InferenceSession(model_file)

    inputs = sess.get_inputs()

    print('inputs')
    for i, e in enumerate(inputs):
        print('  ', i, e.name, e.type, e.shape)

    outputs = sess.get_outputs()
    print('outputs')
    for i, e in enumerate(outputs):
        print('  ', i, e.name, e.type, e.shape)