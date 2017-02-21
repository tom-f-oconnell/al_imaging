# Tom O'Connell
# during rotation in Betty Hong's lab at Caltech, in early 2017

from tom.analysis import *

plt.close('all')

# initialize seaborn, to make its tweaks to matplotlib (should make things look nicer)
# TODO can't figure out how to actually change default linewidth
sns.set_style('darkgrid')
# TODO test. does this mean green is plotted first? (it would seem to)
sns.set_palette('GnBu_d')

'''
Metadata relevant to my PID measurments done to troubleshoot the olfactometer we built on Yuki
Oka's 2-p setup.
'''

# TODO normalized port plot?
# TODO arg to plotting function to normalize subplots or have on same scale

prefix = '/home/tom/lab/hong/tool_validation/pid/oka-2p/'
suffix = '/Episode001.h5'
olfactometer_files = ('SyncData046',
                      '2but1e4001',
                      '2hep',
                      ('2but_e2_atnozzle_500ms',
                       '2but_e2_B_mockA',
                       '2but_e2_C',
                       '2but_e2_D',
                       '2but_e2_E',
                       '2but_e2_F',
                       '2but_e2_H',
                       '2but_e2_I',
                       '2but_e2_J'),
                      ('2but_e2_newvial_redo_A',
                       '2but_e2_newvial_redo_B',
                       '2but_e2_newvial_redo_C',
                       '2but_e2_newvial_redo_D',
                       '2but_e2_newvial_redo_E',
                       '2but_e2_newvial_redo_F',
                       '2but_e2_newvial_redo_H',
                       '2but_e2_newvial_redo_I',
                       '2but_e2_newvial_redo_J001'),
                      '2bute2_atnozzle',
                      '2but_e2_nostoppers_2s',
                      '2but_e2_nostoppers',
                      '2but_e2_nomanifold_withnormallyopenpfo_2s',
                      '2but_e2_nomanifold_withnormallyopenpfo_500ms',
                      '2but_e2_needlestoppers_2s',
                      '2but_e2_needlestoppers_500ms',
                      '2but_e2_needleout',
                      '2but_e2_further_in',
                      'perpendicular',
                      'perpendicular_highpumpspeed_correctflow')

# what was 2pulse_1onset_4scope_16iti_test001 testing?
# the files by the same name as above but minux '001' suffix are trash
# SyncData044 is mostly redundant with SyncData043
# 2but1e4 was a false start to the recording on 2but1e4001
# Note: did not test port G because it did not seem like it would be easy to access

# temporarily omitted:
# 'No crosstalk between neighboring ports' + 'SyncData043'
# tbv6_pin2odor = {5: 'paraffin', 6: '2-butanone 1e-2'}

#olfactometer_files = tuple(map(lambda x: prefix + x + suffix, olfactometer_files))
olfactometer_files = fix_names(prefix, olfactometer_files, suffix)

# TODO make this metadata handled in a dict or something so it is indexed by something
# immediately intelligible?

olf_titles = ('All pins, in order',
              'Different odor vial (2-butanone 1e-4, gain changed)',
              'Different odor vial (2-heptanone 1e-2)',
              ('Responses of different manifold ports, PID inlet at olfactometer outlet', 
                ('A','B','C','D','E','F','H','I','J')),
              ('Manifold ports, redone, this time in opposite order. More attention to needle depth.', 
                ('A','B','C','D','E','F','H','I','J')),
              'Silicon stoppers, PID inlet now near outlet. Better.',
              'No stoppers on 8 unused ports',
              'No stoppers on 8 unused ports',
              'Teflon tubing in place of manifold',
              'Teflon tubing in place of manifold. Questionable.',
              'Stopped needles instead of silicon stoppers',
              'Stopped needles instead of silicon stoppers',
              'Odor needle further out',
              'Odor needle futher in',
              'Perpendicular',
              'Perpendicular, High pump speed')

# TODO fix grouping for n=2 (broken)

# making a list of dictionaries that describes which odors were connected where in each experiment
# used the same vial manually connected to each valve sequentially
tb_all_pin2odor = dict(zip(range(5,12), ['2-butanone 1e-2'] * 7))
most_pin2odor = {5: '2-butanone 1e-2'}

olfactometer_pin2odor = (tb_all_pin2odor, {5: '2-butanone 1e-4'}, {5: '2-heptanone 1e-2'})
olfactometer_pin2odor += (most_pin2odor,) * (len(olfactometer_files) - len(olfactometer_pin2odor))

olf_discard_pre = [0] * len(olfactometer_files)
olf_discard_post = [0] * len(olfactometer_files)

'''
Metadata relevant to the validation of the rearing rigs (for my contribution to Zhannetta's project
'''

# who recorded '20170124_001059.mat' and what is it?
prefix = '/home/tom/lab/hong/tool_validation/pid/rearing-rig/'
rearing_files = ('20170119_123029.mat',
                 '20170121_232120.mat')
rearing_files = tuple(map(lambda x: prefix + x, rearing_files))

# TODO figure out what is up with the br channel (prob. using like 1 trace, spread too small)
rear_titles = ('Back left channel',
               'Back right channel')

rearing_pin2odor = (None,) * len(rearing_files)

rear_discard_pre = [0] * len(rearing_files)
rear_discard_post = [0] * len(rearing_files)

# the gain was changed at this point in this particular trial
rear_discard_post[0] = 0.7

files = olfactometer_files + rearing_files
titles = olf_titles + rear_titles
pin2odor = olfactometer_pin2odor + rearing_pin2odor
discard_pre = olf_discard_pre + rear_discard_pre
discard_post = olf_discard_post + rear_discard_post

for f, t, p2o, dpre, dpost in zip(files, titles, pin2odor, discard_pre, discard_post):
    subtitles = None

    # will then be a tuple (or maybe list)
    if not type(t) is str:
        subtitles = t[1]
        t = t[0]

    process_experiment(f, t, subtitles, secs_before=1, secs_after=3, pin2odor=p2o, \
            discard_pre=dpre, discard_post=dpost)

plt.show()
