import time

class Sleeper():
    def __init__(self, start_time = ( 7, 0, 0),  #  7:00 am
                       end_time   = (22, 0, 0)): # 10:00 pm
        """Sleeps python script during time between start_time and end_time

        Args:
            start_time (tuple, optional): Time to begin sleep in military time. Defaults to ( 7, 0, 0).
            end_time (tuple, optional): Time to stop sleeping in military time. Defaults to (22, 0, 0).
        """
        super(Sleeper, self).__init__()

        # Save start/stop times
        self.start_time = start_time
        self.end_time   = end_time

    def _current_time(self):
        """Get current time in military time

        Returns:
            tuple: hr:min:sec in military time
        """
        return time.localtime()[3: 6]

    def _convert_to_seconds(self, time):
        """Convert military time to seconds

        Args:
            time (tuple): hr:min:sec in military time

        Returns:
            int: time in seconds
        """
        return 3600*time[0] + 60*time[1] + time[0]

    def _is_sleep_time(self):
        """Determint if it is time to sleep

        Returns:
            bool: If True it is time to sleep
        """
        current_time = self._convert_to_seconds(self._current_time())
        start_time   = self._convert_to_seconds(self.start_time)
        end_time     = self._convert_to_seconds(self.end_time)

        return (start_time < current_time) and (current_time < end_time)

    def check(self):
        """Check if its time to sleep, if so sleep!
        """
        current_time = self._convert_to_seconds(self._current_time())
        end_time     = self._convert_to_seconds(self.end_time)

        if self._is_sleep_time():
            print("Fell asleep at ", self._current_time()[0],"\b:" + str(self._current_time()[1]), 
                   "\b. Waking up at ", self.end_time[0],"\b:00.")
            time.sleep(end_time - current_time)
            print("Back to work...")