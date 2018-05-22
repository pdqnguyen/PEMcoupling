"""
Super basic PEM injection info extractor...

To use this:
Copy this script to directory with xml files of interest.
Go to directory and created list of xml names with the name 'xmlfiles.txt'.
Run this script.
Outputs will be saved in 2 .txt files
"""

import os
import subprocess
from optparse import OptionParser
from pemcoupling.getparams import get_times_DTT

parser = OptionParser()
parser.add_option("-f","--directory_from", dest = "directory", 
                  help = "Name of source directory, containing xml files for extracting times.")
parser.add_option("-t","--directory_to", dest= "target",
                  help = "Name of target directory, where channel and times lists will be saved.\
                  If none provided, will be same as -f (directory_from) option.")
parser.add_option("-c","--channels", action = "store_true", dest = "channels", default=False,
                  help = "Output a list of channels found in the DTTs.")
(options, args) = parser.parse_args()
directory = options.directory
target = options.target
# Get source directory
if directory == None:
    print('No source directory specified by -f. Searching for DTTs in current directory.')
    directory = '.'
elif directory[-1] == '/':
    directory = directory[:-1]
# Get target directory
if target == None:
    print('No output directory specified by -t. Output will be saved in source directory.')
    target = directory
elif target[-1] == '/':
    target = target[:-1]
# Search for xml files
if directory == '.':
    fnames = subprocess.check_output(['ls *.xml'], shell=True).splitlines()
else:
    fnames = subprocess.check_output(['ls ' + directory + '/*.xml'], shell=True).splitlines()
# Get times and channels
injection_table, channels = get_times_DTT(fnames, return_channels=True)
# Save injection table to file
with open(target + '/times.txt', 'wb') as times_file:
    times_file.write('name,quiet,injection')
    for row in injection_table:
        times_file.write('\n' + ','.join([str(x) for x in row]))
# Save channel list to file
if options.channels:
    channels = sorted(set(channels))
    with open(target + '/channels.txt', 'wb') as channels_file:
        for c in channels:
            if '_DQ' not in c:
                channels_file.write(c + '_DQ\n')
            else:
                channels_file.write(c + '\n')