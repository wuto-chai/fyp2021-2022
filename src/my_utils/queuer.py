class Queuer:
    def __init__(self, start_frame, position, last_frame, finish_queueing, enter_finish_area_frame=None) -> None:
        self.start_frame = start_frame
        self.position=position
        self.last_frame = last_frame
        self.enter_finish_area_frame = enter_finish_area_frame
        self.finish_queueing = finish_queueing
class PotentialQueuer:
    def __init__(self, start_frame, accumulated_frames) -> None:
        self.start_frame = start_frame
        self.accumulated_frames = accumulated_frames