TODO
- Capture in readme all the stuff
- Run SSP with 7B_8bit / 30B_8bit and observe
- Confirm the distribution (manual check)
- Try sampling with temperature 1 & observing on additions



Later
- generation using custom argmax like in ssp
- test avec un 7/13 sur la machine 4GPUs
    - puis un 7/30 car là le speedup sera p-e présent
    - 2 modes : All-4 GPUs (easy but slow draft) ou 1+3GPUs (draft should then be faster than target, better speedup)
- sujet du batching pour le timing
    - handle batch > 1
    - monter la batch size pour obtenir une vraie diff
    - refactor & clean output: tokens/s for various settings (+ commit)
- experiments on speedup
- handle max context size
- test with a temperature / top_p sampling 
- sampling method as a param
- handle stop token
- clean implementation
- timing issues : pourquoi le manual sampling est plus long que le sampling lib HF?
- debug mem reqs pour voir si monoGPU possible
