# todo 是否只引入run函数，而不引入tokenizer之类的其他函数，到时候让用户自己进入run_eann.py中去调用
from faknow.run.content_based.multimodal.run_eann import run_eann, run_eann_from_yaml
from faknow.run.content_based.multimodal.run_mcan import run_mcan, run_mcan_from_yaml
from faknow.run.content_based.multimodal.run_mfan import run_mfan, run_mfan_from_yaml
