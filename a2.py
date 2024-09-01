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
    command = f'python A_args_PCsampling_demo.py --size={ags[0]}  {aaa} --gap={ags[2]} --gpu {ags[3]} '

    os.system(command)
    return command

nothing=[
    [150, True, 50, 0], [150, False, 50,0],
    # [150, True, 100, 0], [150, False, 100,1],
    # [200, True, 50, 0], [200, False, 50, 0],
]
if __name__ == '__main__':
    with Pool(7) as P:
        list(P.imap(run,nothing))
