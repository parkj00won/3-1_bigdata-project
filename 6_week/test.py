import torch
print(torch.__version__)
print(('gpu' if torch.cuda.is_available() else 'cpu') + ' is available')
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')