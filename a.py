import os
from multiprocessing.pool import Pool
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# print(file_name)
# assert False

def run(ags):
    aaa = ""
    if ags[1]:
        aaa = '--useNet=True'
    command = f'python A_1k_arg_PCsampling_demo.py --size={ags[0]}  {aaa} --gap={ags[2]} --gpu {ags[3]} '

    os.system(command)
    return command

nothing=[
    [400, True, 100, 0], [400, False, 100,1],
    [400,True, 60, 0],[400,False, 60, 0],
    # [400,True, 30, 0],[400,False, 30, 0],
    # [450, True, 100, 0], [450, False, 100, 0],
    # [350, True, 100, 1], [350, False, 100, 1],
]
if __name__ == '__main__':
    with Pool(4) as P:
        list(P.imap(run,nothing))
