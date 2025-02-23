import os
import time
from PIL import Image

from ocr import CompactOCR

if __name__ == "__main__":
    model = CompactOCR(path_model='model/compact_OCR_v1')

    path_test = 'data'
    list_image = os.listdir(path_test)

    t0 = time.time()
    for image_path in list_image:
        img_path = os.path.join(path_test, image_path)

        img = Image.open(img_path).convert('L')
        preds_str = model.process(img)

        print("\nImage: ", image_path)
        print("Predict: ", preds_str)
    t1 = time.time()

    time_one_image = (t1-t0)/len(list_image)
    print('Time cost for one image: ', time_one_image)
    fps = float(1/time_one_image)
    print("FPS = {} ".format(fps, '.1f') )
