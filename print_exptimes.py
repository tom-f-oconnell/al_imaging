#!/usr/bin/env python3

import os
import tom.analysis as ta

image_dirs = [d for d in os.listdir() if ta.is_thorimage_dir(d)]
out = sorted([(ta.get_readable_exptime(d), d) for d in image_dirs], key=lambda x: x[0])

last_day = None
for e in out:
    day = e[0].split('/')[1]
    if last_day is not None and last_day != day:
        print('')
    print(e[0], e[1])
    last_day = day
