# Main11Server Run Commands

## Python executable

Use the Python installation that has `cupy` available:

```powershell
C:\Users\Patrick\AppData\Local\Programs\Python\Python311\python.exe
```

---

## 1) Single server, full field streaming

```powershell
C:\Users\Patrick\AppData\Local\Programs\Python\Python311\python.exe "C:\Users\Patrick\Documents\ActiveCode\FlowWithCurl\FlowStuff\Main11Server\Main_3d.py" --mode server --field density --host 0.0.0.0 --port 5055 --send-every 10
```

---

## 2) Request a small 2D slice from a monitor client

```powershell
C:\Users\Patrick\AppData\Local\Programs\Python\Python311\python.exe "C:\Users\Patrick\Documents\ActiveCode\FlowWithCurl\FlowStuff\Main11Server\monitor_client.py" --host <SERVER_IP> --port 5055 --field density --transfer-mode slice --slice-axis z --slice-index 55 --region -0.5 0.5 -0.5 0.5 -0.5 0.5
```

- `--transfer-mode slice` sends only one 2D plane
- `--slice-axis` can be `x`, `y`, or `z`
- `--slice-index` is the grid index for the slice (`z` uses global Z index)

---

## 3) Request a small cube from a monitor client

```powershell
C:\Users\Patrick\AppData\Local\Programs\Python\Python311\python.exe "C:\Users\Patrick\Documents\ActiveCode\FlowWithCurl\FlowStuff\Main11Server\monitor_client.py" --host <SERVER_IP> --port 5055 --field density --transfer-mode cube --region -0.5 0.5 -0.5 0.5 0.1 0.4
```

- `--transfer-mode cube` sends only the requested 3D subvolume
- `--region` is `XMIN XMAX YMIN YMAX ZMIN ZMAX` in world coordinates

---

## 4) Two-partition NZ split server

### Partition 0 / GPU 0

```powershell
C:\Users\Patrick\AppData\Local\Programs\Python\Python311\python.exe "C:\Users\Patrick\Documents\ActiveCode\FlowWithCurl\FlowStuff\Main11Server\Main_3d.py" --mode server --field density --host 0.0.0.0 --port 5055 --device-id 0 --nz-partitions 2 --partition-index 0 --halo-dir "C:\halo_shared"
```

### Partition 1 / GPU 1

```powershell
C:\Users\Patrick\AppData\Local\Programs\Python\Python311\python.exe "C:\Users\Patrick\Documents\ActiveCode\FlowWithCurl\FlowStuff\Main11Server\Main_3d.py" --mode server --field density --host 0.0.0.0 --port 5056 --device-id 1 --nz-partitions 2 --partition-index 1 --halo-dir "C:\halo_shared"
```

Notes:
- both partitions must use the **same** `--halo-dir`
- connect the monitor client to the partition/port you want to inspect

---

## 5) Request a small cube from only slab 1

```powershell
C:\Users\Patrick\AppData\Local\Programs\Python\Python311\python.exe "C:\Users\Patrick\Documents\ActiveCode\FlowWithCurl\FlowStuff\Main11Server\monitor_client.py" --host <SERVER_IP> --port 5056 --field density --transfer-mode cube --region -0.5 0.5 -0.5 0.5 0.1 0.4 --no-plot
```

---

## 6) Stats-only mode

Add `--no-plot` to the client command:

```powershell
C:\Users\Patrick\AppData\Local\Programs\Python\Python311\python.exe "C:\Users\Patrick\Documents\ActiveCode\FlowWithCurl\FlowStuff\Main11Server\monitor_client.py" --host <SERVER_IP> --port 5055 --field density --transfer-mode slice --slice-axis z --slice-index 55 --region -0.5 0.5 -0.5 0.5 -0.5 0.5 --no-plot
```

---

## Common options

### Server options
- `--field density|density2|flow|curl`
- `--send-every N`
- `--sim-ups N`
- `--max-steps N`
- `--nz-partitions N`
- `--partition-index I`
- `--halo-dir <path>`

### Client options
- `--transfer-mode cube|slice`
- `--slice-axis x|y|z`
- `--slice-index N`
- `--region XMIN XMAX YMIN YMAX ZMIN ZMAX`
- `--max-points N`
- `--no-plot`
