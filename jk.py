# gevent
#from gevent import monkey
#from gevent.pywsgi import WSGIServer
#monkey.patch_all()

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import datasets, transforms
from flask import Flask, request
import os
import traceback
import time
import json
import logging
from PIL import ImageFile
from MyLogger import get_logger_Rotating, get_logger_Rotating_feedback
import requests
from io import BytesIO

ImageFile.LOAD_TRUNCATED_IMAGES = True

# zk服务注册
#from ZooKeeper import ZooKeeper
#envs = os.environ
#zk = ZooKeeper(config='zk_config.yml', stage=envs["ZKSTAGE"])
#zk.register()   # 即可完成服务http://ip:port的注册


# Get dictionary of the category
def get_key(dct, value):
    return [k for (k, v) in dct.items() if v == value]


# Define the arguments ###
data_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def Predict_class(image_data):
    inputs = data_transforms(image_data)
    inputs.unsqueeze_(0)
    
    inputs = Variable(inputs.cuda())      # Use GPU
    #inputs = Variable(inputs)            # Use CPU only
    
    outputs = model(inputs)              # Forward compute the results according to the model arguments
    _, preds = torch.max(outputs.data, 1)
    class_name = get_key(mapping, preds.item())    # Obtain category of the picture
    return class_name


app = Flask(__name__)


@app.route('/check_alive', methods=['GET'])
def check_alive():
    return {"alive": True}, 200


@app.route('/Material_classify/v1', methods=['POST'])
def Material_classify():
    pid = os.getpid()
    response = {"pid": pid, "ret":1, "msg":"invalid parameter"}  # 默认的请求返回格式
    if request.method == "POST":
        if request.headers['Content-Type'] == 'application/json':
            try:
                request_dict = json.loads(request.get_data())
                request_params = request_dict
                imgURL = request_params["contents"]

            except:
                return response, 400
            
            else:
                if not imgURL:
                    #return response, 400
                    response["ret"] = 0
                    response["msg"] = "success"
                    response["data"] = []
                    response_data = response["data"]
                    return response

            response["ret"] = 0
            response["msg"] = "success"
            #response["total"] = total
            #response["successTotal"] = 0
            response["data"] = []
            response_data = response["data"]

            for url_item in imgURL:
                url = url_item["url"]
                materialid = url_item["materialid"]
                toReturn_temp = dict()
                toReturn_temp["url"] = url
                toReturn_temp["materialid"] = materialid
                
                try:
                    img = requests.get(url)

                except exceptions.Timeout:
                    toReturn_temp["msg"] = "time out"
                    toReturn_temp["ret"] = 2
                    response_data.append(toReturn_temp)
                    continue
                except exceptions.HTTPError:
                    toReturn_temp["msg"] = "invalid parameter"
                    toReturn_temp["ret"] = 1
                    response_data.append(toReturn_temp)
                    continue
                except exceptions.InvalidURL:
                    toReturn_temp["msg"] = "invalid parameter"
                    toReturn_temp["ret"] = 1
                    response_data.append(toReturn_temp)
                    continue
                except exceptions.MissingSchema:
                    toReturn_temp["msg"] = "invalid parameter"
                    toReturn_temp["ret"] = 1
                    response_data.append(toReturn_temp)
                    continue
                except exceptions.ConnectionError:
                    toReturn_temp["msg"] = "invalid parameter"
                    toReturn_temp["ret"] = 1
                    response_data.append(toReturn_temp)
                    continue

                else:
                    if (img.status_code == 400 or img.status_code == 401 or img.status_code == 402 or
                        img.status_code == 403 or img.status_code == 404 or img.status_code == 405):
                        toReturn_temp["msg"] = "invalid parameter"
                        toReturn_temp["ret"] = 1
                        response_data.append(toReturn_temp)
                        continue
                    
                    else:
                        try:
                            image_data = Image.open(BytesIO(img.content))
                            image_data = image_data.convert("RGB")
                            ### results ###
                            categories = Predict_class(image_data)    ### Achieve category of the picture
                            
                            if categories == [] or categories == None:
                                toReturn_temp["msg"] = "success"
                                toReturn_temp["ret"] = 0        
                                toReturn_temp["category_name"] = "others"

                                #log_info = "%s:%s"%(url, category_name)
                               # info_logger.info(log_info)                            
                                response_data.append(toReturn_temp) 

                            else:
                                items = categories[0].split('-')
                                category_name = items[0:1]
                                category_id = items[1:2]
                                category_classify = items[2:3]
                                category_classify_id = items[3:4]
                                
                                toReturn_temp["msg"] = "success"
                                toReturn_temp["ret"] = 0
                                toReturn_temp["category_name"] = str(category_name)
                                toReturn_temp["category_id"] = str(category_id)
                                toReturn_temp["category_classify"] = str(category_classify)
                                toReturn_temp["category_classify_id"] = str(category_classify_id)
                            
                                #response["successTotal"] += 1           # 成功识别一张URL并返回正确结果
                                log_info = "%s:%s"%(url, category_name)
                                info_logger.info(log_info)
                                response_data.append(toReturn_temp)
                        except:
                            print(traceback.format_exc())
                            toReturn_temp["msg"] = "MCC failed"
                            toReturn_temp["ret"] = 3
                            response_data.append(toReturn_temp)

        elif request.headers['Content-Type'] == 'application/octet-stream':
            try:
                img_stream = request.data
            except:
                return response, 400
            
            # trying to read image data
            try:
                image_data = Image.open(BytesIO(img_stream))
                image_data = image_data.convert("RGB")
            except:
                response["ret"] = 2
                response["msg"] = "invalid Content-Type"
                return response
            
            # 请求正确，开始初始化返回结果
            response["ret"] = 0
            response["msg"] = "success"
            response["data"] = dict()
            result = response["data"]
            
            try:
                categories = Predict_class(image_data)

                if categories == [] or categories == None:
                    result["msg"] = "success"
                    result["ret"] = 0
                    result["category_name"] = "others"
                else:
                    items = categories[0].split('-')
                    category_name = items[0:1]
                    category_id = items[1:2]
                    category_classify = items[2:3]
                    category_classify_id = items[3:4]
                
                    result["msg"] = "success"
                    result["ret"] = 0
                    result["category_name"] = str(category_name)
                    result["category_id"] = str(category_id)
                    result["category_classify"] = str(category_classify)
                    result["category_classify_id"] = str(category_classify_id)
            except:
                print(traceback.format_exc())
                result["msg"] = "MCC failed"
                result["ret"] = 1
        else:
            response = {"ret": 2, "msg": "invalid Content-Type"}
            return response
    else:
        return response, 400

    return response, 200

if __name__ == "__main__":
    info_logger = get_logger_Rotating(log_filename='classify_results',
            maxBytes=12 * 1024 * 1024,  # 每个日志文件12MB,大约每个文件能保存10万条预测结果
            backupCount=100,
            level=logging.INFO)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
    
    with open("./config.json", "r") as temp:
        CONFIGS = json.loads(temp.read())
    
    model_path = CONFIGS["model_path"]
    label_path = os.path.join(model_path, CONFIGS["label_path"])
    pth_path = os.path.join(model_path, CONFIGS["pth_path"])

    mapping = eval(open(label_path).read())

    ### Load model ###
    modelft_file = os.path.join(model_path, CONFIGS["pth_path"])
    model = torch.load(modelft_file).cuda()  
    #model = torch.load(modelft_file, map_location=torch.device('cpu'))
    model.eval()
    
    app.run(host='0.0.0.0',port=8080,debug=True)
    #url = '0.0.0.0'
    #port = 5000
    #print('App is running now on docker with http://%s:%d/'%(url, port))
    #http_server = WSGIServer((url, port), app)
    #http_server.serve_forever()
