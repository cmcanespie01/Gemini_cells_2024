import os
import numpy as np
import re
from ..DAQ import DAQ

class GeminiMirage(DAQ):
    """Interface layer for Gemini, using Mirage
    """

    __version = 0.1
    __name = 'GeminiMirage'
    __authors = ['Brendan Kettle']

    def __init__(self, exp_obj, options=None):
        """Initiate parent base DAQ class to get all shared attributes and funcs"""
        super().__init__(exp_obj, options=options)
        return

    def _build_shot_filepath(self, diagnostic, date, run, shotnum, ext):
        """This is used internally, and so can be DAQ specific"""
        if 'shotnum_zfill' in self.options:
            zfill = self.options['shotnum_zfill']
        else:
            zfill = 3 # default?
        # check file?
        shot_path = f'{self.data_folder}/{diagnostic}/{date}/{run}/Shot{str(shotnum).zfill(zfill)}.{ext}'
        return shot_path
    
    def _shot_dict_from_GSN(self, GSN):
        """Internal function for getting a date / run / shot from a GSN"""
        if 'GSN_shot_dicts' not in self.config['DAQ_config']:
            print('DAQ error; shot_dict_from_GSN(); no GSN_shot_dicts filepath set in DAQ config')
            return None
        lookup = self.load_file(self.config['DAQ_config'])
        
        shot_dict = {}
        return shot_dict
    
    def _laser_energy_from_GSN(self, GSN):
        # look up eCat file
        return

    def build_time_point(self, shot_dict):
        """Universal function to return a point in time for DAQ, for comparison, say in calibrations
        """
        # for Gemini, use date / run / shot
        date_str = str(shot_dict['date'])
        year = int(date_str[0:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        if 'run' in shot_dict and shot_dict['run']:
            run_str = shot_dict['run']
            m = re.search(r'\d+$', run_str) # gets last numbers
            run = int(m.group())
        else:
            run = 0
        if 'shotnum' in shot_dict:
            shotnum = shot_dict['shotnum']
        else:
            shotnum = 0

        # weight the different components to make a unique increasing number?
        time_point = year*1e10 + month*1e8 + day*1e6 + run*1000 + shotnum
        return  time_point
    
    # perhaps some of this can move to base class?
    def get_shot_data(self, diag_name, shot_dict):
        """Univeral function for returning shot data given a diagnostic and shot dictionary
        """

        # Check if shot_dict is not dictionary; could just be filepath
        if isinstance(shot_dict, str):
            # If not dictionary, assume filepath
            segs = shot_dict.split('/')
            shot_str = segs[-1].split('.')[0]
            run_str = segs[-2]
            date_str = segs[-3]
            # print(shot_str)
            # print(run_str)
            # print(date_str)
            shot_filepath = f'{self.data_folder}/{shot_dict}'
            shot_data = self.load_imdata(shot_filepath)
        else:
            diag_config = self.ex.diags[diag_name].config
            required = ['data_folder','data_ext','data_type']
            for param in required:
                if param not in diag_config:
                    print(f"get_shot_data() error: {self.__name} DAQ requires a config parameter '{param}' for {diag_name}")
                    return None
                
            # TO DO: OR can use GSN?
            if 'GSN' in shot_dict:
                shot_dict = self._shot_dict_from_GSN(shot_dict['GSN'])
            
            required = ['date','run','shotnum']
            for param in required:
                if param not in shot_dict:
                    print(f"get_shot_data() error: {self.__name} DAQ requires a shot_dict['{param}'] value")
                    return None

            shot_filepath = self._build_shot_filepath(diag_config['data_folder'], shot_dict['date'], shot_dict['run'], shot_dict['shotnum'], diag_config['data_ext'])

            if diag_config['data_type'] == 'image':
                shot_data = self.load_imdata(shot_filepath)
            else:
                print('Non-image data loading not yet supported... probably need to add text at least?')

        return shot_data
    
    def get_shot_dicts(self, diag_name, timeframe, exceptions=None):
        """timeframe can be 'all' or a dictionary containing lists of dates, or runs"""

        diag_config = self.ex.diags[diag_name].config
        diag_folder = f"{self.data_folder}/{diag_config['data_folder']}"

        shot_dicts = []

        # scan all folders?
        if isinstance(timeframe, str) and timeframe.lower() == 'all':
            # get date folders
            dates = []
            for dir_name in os.listdir(diag_folder):
                if os.path.isdir(os.path.join(diag_folder, dir_name)):
                    # add filename to list (quick bodge here to try and catch only real date folders)
                    if len(dir_name) == 8:
                        dates.append(int(dir_name))
        elif isinstance(timeframe, dict) and 'dates' in timeframe:
            dates = timeframe['dates']
        elif isinstance(timeframe, dict) and 'date' in timeframe:
            dates = [timeframe['date']]

        # now that we have dates, for each, get run(s)
        for date in sorted(dates):
            date_folder = os.path.join(diag_folder, str(date))
            # runs passed
            if isinstance(timeframe, dict) and 'runs' in timeframe:
                runs = timeframe['runs']
            # single run
            elif isinstance(timeframe, dict) and 'run' in timeframe:
                runs = [timeframe['run']]
            # scan folder
            else:
                runs = []
                for run_name in os.listdir(date_folder):
                    if os.path.isdir(os.path.join(date_folder, run_name)):
                        runs.append(run_name)
            # now we have date and runs, get shots
            for run in sorted(runs):
                run_folder = os.path.join(date_folder, str(run))
                shotnums = []
                if isinstance(timeframe, dict) and 'shotnums' in timeframe:
                    shotnums = timeframe['shotnums']
                else:
                    for filename in os.listdir(run_folder):
                        if os.path.isfile(os.path.join(run_folder, filename)):
                            if 'shot' in filename.lower():
                                m = re.search(r'\d+$', os.path.splitext(filename)[0]) # gets last numbers, after extension removed
                                shotnums.append(int(m.group()))
                                #print(f"{date} / {run} / {shotnums[-1]}")
                    shotnums = sorted(shotnums)

                # OK, build the list to return!
                for shotnum in shotnums:
                    shot_dicts.append({'date': date, 'run': run, 'shotnum': shotnum})

        return shot_dicts

