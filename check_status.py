import torch

try:
    chk = torch.load('/home/atikr/random/PIGNN/runs/pignn/pignn_checkpoint.pt', map_location='cpu')
    print('Checkpoint Epoch:', chk.get('epoch', 'N/A'))
    print('Best Validation Loss:', chk.get('best_loss', 'N/A'))
    
    # Print the last few history losses if they exist in the checkpoint
    history = chk.get('history', {})
    if 'total' in history and len(history['total']) > 0:
        print('History Length:', len(history['total']))
        print('Last Saved Total Loss:', history['total'][-1])
        print('Last Saved FVM Loss:', history.get('fvm', [])[-1])
        print('Last Saved BTC Loss:', history.get('btc', [])[-1])
        print('Last Saved BTC Stage Error (m):', history.get('btc_stage_err_m', [])[-1])
    else:
        print('No history entries in checkpoint yet.')
except Exception as e:
    print('Error loading checkpoint:', e)
