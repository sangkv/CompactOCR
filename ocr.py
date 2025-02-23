import torch
from torchvision import transforms
from PIL import Image

class ResizeNormalize(object):
    def __init__(self, imgH=32, interpolation=Image.BICUBIC):
        self.imgH = imgH
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        (w, h) = img.size
        imgW = int(w * (self.imgH / h))
        img = img.resize((imgW, self.imgH), self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img.unsqueeze(0)

class CTCDecode(object):
    def __init__(self, character):
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i + 1

        self.character = ['[CTCblank]'] + dict_character

    def decode(self, text_index, length):
        texts = []
        for index, l in enumerate(length):
            t = text_index[index, :]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
        return texts

class CompactOCR:
    def __init__(self, path_model):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.jit.load(path_model).to(self.device).eval()
        self.transforms = ResizeNormalize(imgH=32)
        self.list_of_characters = r"""aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ """
        self.converter = CTCDecode(self.list_of_characters)
    
    def process(self, img):
        with torch.no_grad():
            image_tensors = self.transforms(img)
            batch_size = image_tensors.size(0)
            image_tensors = image_tensors.to(self.device)
            preds = self.model(image_tensors)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            _, preds_index = preds.max(2)
            preds_str = self.converter.decode(preds_index, preds_size)

            return preds_str
