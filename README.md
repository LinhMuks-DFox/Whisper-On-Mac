# Whisper-On-Mac
A temporary solution of using openAI-whisper on mac

Using the `whisper *.mp3 -f txt --device mps` command on a Mac (on M1+ series chips) returned to cause the following error because the backend of the mac's mps does not support some of the operations of the sparse tensor for the time being:
```
NotImplementedError: Could not run ‘aten::empty.memory_format’ with arguments from the ‘SparseMPS’ backend.
```
Never mind, this project helps you to solve this problem, the principle is simple: convert the model to a dense form (maybe at the expense of memory efficiency), then convert it to the mps device and transcribe it.

