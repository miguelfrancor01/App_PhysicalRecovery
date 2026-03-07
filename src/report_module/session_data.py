"""
Session data container.

Stores statistics of the exercise session.
"""


class SessionData:

    def __init__(self):

        self.exercise_name = "Front Raise"

        # datos simulados por ahora
        self.total_reps = 12
        self.correct_reps = 10
        self.incorrect_reps = 2

        self.duration_seconds = 45
        self.avg_fps = 28

    def get_summary(self):

        return {
            "exercise": self.exercise_name,
            "total_reps": self.total_reps,
            "correct_reps": self.correct_reps,
            "incorrect_reps": self.incorrect_reps,
            "duration": self.duration_seconds,
            "avg_fps": self.avg_fps
        }