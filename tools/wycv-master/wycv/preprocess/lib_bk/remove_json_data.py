from utils import *

def remove_json_data(json_folder_path,json_out_path):

    print('Loading files...')
    for files in os.listdir(json_folder_path):
        if not files.endswith('.json'): continue
        instance=json_to_instance(os.path.join(json_folder_path,files))
        # print(instance["imageData"])
        instance["imageData"]=None
        print(instance["imageData"])
        if not os.path.exists(json_out_path): os.makedirs(json_out_path)
        instance_to_json(instance, os.path.join(json_out_path,files))

if __name__=='__main__':
    remove_json_data(json_folder_path=r'G:\huawei_jianpan\qingxi\20210112',json_out_path=r'G:\huawei_jianpan\qingxi\test')