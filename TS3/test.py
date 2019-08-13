from models import cpm_vgg16, hourglass
from models import count_network_param, get_model_infos

student_cpm_config = {'stages': 3,
                      'dilation': [1],
                      'pooling' : [1, 1, 1,],
                      'downsample': 8,
                      'argmax'  : 4,
                      'pretrained': False}

student_cpm = cpm_vgg16(student_cpm_config, 68)
#print('CPM:\n{:}'.format(student_cpm))
FLOPs, _    = get_model_infos(student_cpm, (1, 3, 64, 64))
print('CPM-Parameters : {:} MB, FLOP : {:} MB.'.format(count_network_param(student_cpm) / 1e6, FLOPs))

student_hg_config  = {'nStack'  : 4,
                      'nModules': 2,
                      'nFeats'  : 256,
                      'downsample' : 4}

student_hg  = hourglass(student_hg_config, 68)
FLOPs, _    = get_model_infos(student_hg, (1, 3, 64, 64))
#print('CPM:\n{:}'.format(student_cpm))
print('HG--Parameters : {:} MB, FLOP : {:} MB.'.format(count_network_param(student_hg ) / 1e6, FLOPs))
