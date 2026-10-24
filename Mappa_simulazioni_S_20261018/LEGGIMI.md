# Mappa delle simulazioni S già accettate

Il file compatto `006_reload_ss_1_S.analysis.pkl.gz` caricato il 18 ottobre
conserva i profili, ma ha rimosso le tracce sintetiche e i checkpoint. Non permette
di costruire qui una mappa fedele delle simulazioni. Non sono state utilizzate
forme d'onda inventate o tracce ricavate dal solo integrale cinematico.

Il codice della run salva invece la traccia sintetica accettata nel **pickle
originale completo**, nel campo
`local_continuation_state.simulation_attributes.synthetic_waveform`.
Questo script recupera quelle tracce, già comprensive della scala di ampiezza
accettata, e le confronta con le onde registrate caricate dal principale.
Non esegue forward, inversioni o iterazioni del gradiente.

## Esecuzione

1. Copia `crea_mappa_simulazioni.py` nella cartella che contiene
   `local_optimization_FWI_velocity_1D.py`.
2. Mantieni nel principale le stesse impostazioni di preprocessing della run
   S: banda, durata, eventuale downsampling, canale e sorgente.
3. Dalla cartella del progetto, nel suo ambiente Python, esegui:

```bash
poetry run python crea_mappa_simulazioni.py \
  --main local_optimization_FWI_velocity_1D.py \
  --results /percorso/ai/risultati/006_reload_ss_1_S.pkl
```

Sostituisci soltanto il percorso del `.pkl`: è quello originale nella cartella
della run, non il file `.analysis.pkl.gz`. Se normalmente lanci il progetto con
`python3` anziché Poetry, usa lo stesso interprete anche per questo comando.

Non serve installare una patch né modificare il principale. Lo script ne
esegue una copia in memoria, sostituendo `process_uw_file` con l'esportazione e
disabilitando i metodi forward/inversione. Seleziona la polarizzazione S e il
nome del file richiesto; riutilizza il caricamento e il preprocessing esistenti.
Il caricamento delle onde originali richiede la memoria normalmente usata dalla
run. La lettura del pickle è sequenziale, anche per le righe poi escluse.

I risultati vengono scritti nella nuova cartella
`006_reload_ss_1_S_mappe_simulazioni`, accanto al pickle. Puoi scegliere un'altra
cartella con `--output /percorso/cartella`.

## Figure e dati prodotti

- `mappa_simulazioni.png`: sole tracce simulate accettate, finestra 20–25 µs.
- `confronto_stesse_onde.png`: dati e simulazioni agli stessi istanti invertiti,
  residui e stress di taglio.
- `confronto_tutte_le_onde.png`: tutte le onde registrate del tratto, confrontate
  con le simulazioni disponibili e con lo stress.
- `residuo_scala_propria.png`: residuo con scala di colore separata, esplicitamente
  amplificata per renderlo leggibile.
- `tracce_per_confronto.npz`: file compatto, senza checkpoint, con tutte le
  tracce necessarie a riprendere il confronto. **È questo il file da passarmi**:
  evita di caricare nuovamente il pickle completo. Contiene le onde complete
  agli istanti invertiti e il diretto 20–25 µs di tutte le onde del tratto.
- `scarti_finestra_20_25us.csv`: norma quadratica relativa del residuo nel solo
  intervallo grafico. Questa norma non viene chiamata misfit originale, perché
  la finestra della run può essere diversa.
- `provenienza.json`: file/assi utilizzati, hash del principale, clock,
  dimensioni e convenzioni del grafico.

## Convenzioni che rendono il confronto interpretabile

**Ampiezza.** Dati, simulazioni e residui nei confronti principali condividono
la stessa scala di colore, ricavata dalle ampiezze mostrate. Non si aggiungono
fattori di scala, normalizzazioni per onda o riallineamenti temporali. La scala
già adattata durante l'inversione rimane quella conservata nel checkpoint.
Questo confronto non costituisce una nuova calibrazione di ampiezza assoluta.

**Campionamento temporale.** Si recuperano soltanto le inversioni effettive,
non le copie dei checkpoint delle onde riutilizzate. Nella run compatta ci sono
117 inversioni su 12.866 onde. Per le tracce rade si mantiene graficamente
l'ultima traccia disponibile **dal suo istante fino alla successiva**; le tacche
nere segnano le inversioni. Questa convenzione evita di anticipare visivamente
una transizione spostando il bordo a metà fra campioni lontani. Il mantenimento
non rappresenta un nuovo forward nell'intervallo e non verifica l'effetto della
geometria delle onde saltate. Non vengono interpolate linearmente le forme
d'onda. Lo stesso mantenimento vale per il pannello dei dati radi e dei residui;
il pannello delle onde registrate dense mostra invece ciò che accade anche
nei periodi senza inversioni. Nessuna stima dell'onset entro una lacuna va
ricavata dalla sola mappa simulata.

**Clock.** Si applica l'ipotesi di rescaling già concordata, senza aggiungere
parametri: gli estremi dell'asse UW sono associati ai due sync meccanici del
file, letti dal principale. Si trasforma soltanto il tempo sperimentale delle
onde; l'asse acustico in µs e la curva meccanica restano invariati. Il fattore
non viene scelto per massimizzare la correlazione con lo stress. L'ipotesi
fisica sul clock resta da verificare indipendentemente.

**Copertura.** La mappa si limita alla selezione che possiede risultati salvati,
non all'intera acquisizione di circa 1.850 s mostrata nella figura originale.
Il terzo piccolo rilascio rimane parzialmente osservato.

## Verifiche effettuate e limite attuale

Sono stati verificati: esclusione delle onde riutilizzate, recupero della
traccia senza applicare due volte la scala, rifiuto di checkpoint mancanti o
troncati, indici duplicati, assi irregolari, salvataggio senza alterare le
tracce, disabilitazione dell'avvio originale delle inversioni e generazione
delle figure su una fixture sintetica. La trasformazione del principale è
stata compilata anche sulla versione `(8).py` caricata in questa conversazione.

**Non è stato possibile eseguire il confronto sperimentale qui:** mancano le
tracce rimosse dall'esportazione compatta e i dati UW originali preprocessati.
La verifica locale non sostituisce l'esecuzione nel progetto originale. Lo
script si ferma con un messaggio esplicito se il formato del principale, gli
indici o le dimensioni non corrispondono; non sostituisce i dati mancanti.

Per ripetere le verifiche autonome:

```bash
python3 verifica_mappa.py
```
