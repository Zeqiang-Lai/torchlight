from tqdm.auto import tqdm
import time

def multiline_tqdm(*args, **kwargs):
    kwargs = dict(bar_format="{l_bar}{bar}|{n_fmt}/{total_fmt}", **kwargs)
    with tqdm(*args, **kwargs) as line1:
        with tqdm(total=len(line1), bar_format="[{elapsed}<{remaining}, {rate_fmt}{postfix}]") as line2:
            for i in line1:
                yield i
                line2.update()


I = list(range(int(1e6)))

# for i in tqdm(I):  # oneline
    # time.sleep(0.01)

for i in multiline_tqdm(I):
    time.sleep(0.01)