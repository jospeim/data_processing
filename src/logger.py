# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 09:22:10 2026

@author: joachim.eimery
"""

from datetime import datetime
from pathlib import Path


def create_result_file(base_dir: str = "results") -> Path:
    """
    Crée un fichier de résultats horodaté et retourne son chemin.
    """
    Path(base_dir).mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_path = Path(base_dir) / f"run_{timestamp}.txt"

    file_path.touch()
    return file_path


def write_result(file_path: Path, message: str) -> None:
    """
    Ajoute un message dans le fichier de résultats.
    """
    with file_path.open("a", encoding="utf-8") as f:
        f.write(message + "\n")
