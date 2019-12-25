from google_images_download import google_images_download
import json


respon = google_images_download.googleimagesdownload()

rule_config_path = './garbage_classify_rule.json'

config = open(rule_config_path, 'r')
json_data = json.load(config)
print(json_data)

for index, values in json_data.items():
    # if int(index) <= 9:
    #     continue
    # print(item, values)
    keyword = values.split('/')[1]
    # print(keyword)

    arguments = {
        "keywords": keyword,
        "limit": 200,
        "size": ">400*300",
        "image_directory": str(index),
        "chromedriver": "./chromedriver.exe"
    }

    respon.download(arguments)
# paths = respon.download(arguments)
