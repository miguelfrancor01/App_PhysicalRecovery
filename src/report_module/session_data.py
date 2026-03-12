"""Módulo contenedor de datos de sesión de ejercicio.

Almacena y expone las estadísticas recopiladas durante una sesión
de análisis biomecánico.
"""


class SessionData:
    """Contenedor de estadísticas de una sesión de ejercicio.

    Almacena los datos relevantes de la sesión como el nombre del
    ejercicio, repeticiones, duración y FPS promedio, y los expone
    a través de un resumen en forma de diccionario.
    """

    def __init__(self):
        """Inicializa SessionData con valores por defecto.

        Los datos de repeticiones y duración son simulados hasta que
        se integre la lógica de captura real desde el pipeline.
        """
        self.exercise_name = "Front Raise"

        # datos simulados por ahora
        self.total_reps = 12
        self.correct_reps = 10
        self.incorrect_reps = 2

        self.duration_seconds = 45
        self.avg_fps = 28

    def get_summary(self):
        """Retorna un resumen de la sesión como diccionario.

        Returns
        -------
            dict: Diccionario con las claves:
                - 'exercise' (str): Nombre del ejercicio.
                - 'total_reps' (int): Total de repeticiones detectadas.
                - 'correct_reps' (int): Repeticiones correctas.
                - 'incorrect_reps' (int): Repeticiones incorrectas.
                - 'duration' (int): Duración de la sesión en segundos.
                - 'avg_fps' (float): FPS promedio durante la sesión.

        """
        return {
            "exercise": self.exercise_name,
            "total_reps": self.total_reps,
            "correct_reps": self.correct_reps,
            "incorrect_reps": self.incorrect_reps,
            "duration": self.duration_seconds,
            "avg_fps": self.avg_fps,
        }
