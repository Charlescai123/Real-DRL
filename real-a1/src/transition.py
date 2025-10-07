import pickle


class TrajectorySegment:
    def __init__(self):
        self.state = [0.0] * 12
        self.last_action = [0.0] * 6
        self.failed = False
        self.normal_operation = True
        self.sequence_number = 0
        self.observations = []
        self.student_activate = True

    @staticmethod
    def pickle_load_pack(packet):
        s = TrajectorySegment()
        seg = pickle.loads(packet)  # observations, last_action, failed, operations_mode, student_activate_mode
        s.observations = seg[0]
        s.last_action = seg[1]
        s.failed = seg[2]
        s.normal_operation = seg[3]
        s.sequence_number = seg[4]
        s.student_activate = seg[5]
        return s
