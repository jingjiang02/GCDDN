import numpy as np


def check_best(path, name='DensePASS19'):
    mious = []
    mious_teacher = []
    mious_fusion = []
    try:
        with open(path, mode='r', encoding='utf-8') as f:
            for line in f.readlines():
                if line.strip() == '':
                    continue
                if line.find(f'[TEST {name} END]') != -1:
                    miou = float(line.split('mIoU:')[-1].strip())
                    mious.append(miou)
                if line.find(f'[TEST {name}_TEACHER END]') != -1:
                    miou = float(line.split('mIoU:')[-1].strip())
                    mious_teacher.append(miou)
                if line.find(f'[TEST {name}_FUSION END]') != -1:
                    miou = float(line.split('mIoU:')[-1].strip())
                    mious_fusion.append(miou)
    except Exception:
        print(f'No such file in [{path}]!')
        print('-' * 50)
        return
    if len(mious_teacher) != 0:
        gap = len(mious) - len(mious_teacher)
        print(
            f'TEACHER best miou={max(mious_teacher)}, best epoch={np.argmax(mious_teacher) + gap + 1}, now miou={mious_teacher[-1]}, now epoch={len(mious_teacher) + gap}')
    if len(mious_fusion) != 0:
        print(
            f'Fusion best miou={max(mious_fusion)}, best epoch={np.argmax(mious_fusion) + 1}, now miou={mious_fusion[-1]}, now epoch={len(mious_fusion)}')

    if len(mious) != 0:
        print(
            f'best miou={max(mious)}, best epoch={np.argmax(mious) + 1}, now miou={mious[-1]}, now epoch={len(mious)}')
    else:
        print(f'no miou find, check path [{path}]!')
    print('-' * 50)


if __name__ == '__main__':
    root_dir = '/home/jjiang/experiments/GCDDN/workdirs/cityscapes'

    print('***[0]outdoor pretrain***')
    work_dir = 'trans4pass_plus_small_512x512_norm'
    file_name = '2024-12-21-12-27_Trans4PASS_trans4pass_v2_cityscape_log.txt'
    check_best(
        f'{root_dir}/{work_dir}/{file_name}'
    )
