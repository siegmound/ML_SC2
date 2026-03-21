from pathlib import Path

import pandas as pd
import sc2reader
from sc2_bridge_v3_counter120 import SC2Bridge
from project_paths import dataset_path, replay_path
from joblib import Parallel, delayed
from tqdm import tqdm

OUTPUT_FILE = dataset_path("starcraft_full_dataset_v3_counter120.csv")
REPLAY_FOLDER = replay_path()
N_JOBS = 8
MIN_DURATION_SEC = 180
MAX_DURATION_SEC = 1800
DENOISE_CUTOFF_SEC = 120


def worker_process(f_path):
    """Processa un singolo replay in un worker separato."""
    try:
        bridge = SC2Bridge()
        replay_header = sc2reader.load_replay(f_path, load_level=0)
        duration_sec = replay_header.length.seconds

        if duration_sec < MIN_DURATION_SEC or duration_sec > MAX_DURATION_SEC:
            return None

        df = bridge.process_single_replay(f_path)
        if df is not None and not df.empty:
            return df[df['time_sec'] > DENOISE_CUTOFF_SEC]

    except Exception as e:
        return f"ERROR_DETAILS: {str(e)}"

    return None


def main():
    replay_dir = Path(REPLAY_FOLDER)
    if not replay_dir.exists():
        print(f"ERRORE: La cartella {REPLAY_FOLDER} non esiste!")
        return

    files = [
        str(replay_dir / f)
        for f in replay_dir.iterdir()
        if f.name.endswith('.SC2Replay')
    ]

    if not files:
        print("ERRORE: Nessun file .SC2Replay trovato!")
        return

    print(f"Avvio estrazione massiva su {len(files)} replay...")
    print(
        f"Configurazione: {N_JOBS} core | Denoising > {DENOISE_CUTOFF_SEC}s | "
        f"Parser: v3 counter+120s"
    )

    print("Verifica integrità primo file...")
    test_res = worker_process(files[0])
    if isinstance(test_res, str) and "ERROR_DETAILS" in test_res:
        print("--- CRASH RILEVATO NEL WORKER ---")
        print(test_res)
        print("---------------------------------")
        return

    results = Parallel(n_jobs=N_JOBS)(
        delayed(worker_process)(f) for f in tqdm(files, desc="Estrazione")
    )

    print("Concatenazione dei dati in corso...")
    all_frames = []
    errors_count = 0

    for r in results:
        if isinstance(r, pd.DataFrame):
            all_frames.append(r)
        elif isinstance(r, str) and "ERROR" in r:
            errors_count += 1

    if all_frames:
        final_df = pd.concat(all_frames, ignore_index=True)
        final_df.to_csv(OUTPUT_FILE, index=False)
        print("OPERAZIONE COMPLETATA!")
        print(f"Dataset creato: {OUTPUT_FILE}")
        print(f"Snapshot totali: {len(final_df)}")
        print(f"File saltati per errore/corruzione: {errors_count}")
    else:
        print("ERRORE: Nessun dato estratto nonostante il test iniziale sia passato.")


if __name__ == "__main__":
    main()
