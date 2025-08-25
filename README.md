# OldNeuralNets
Repo for my HS Neural Nets Project &amp; my attempts at rewriting in C++. More info further down.

# TODO:
1. Update documentation for updated code (In progress)
2. Test N-layer functionality of existing code
3. Check for and remedy memory leaks

# Overview:
This contains code for a simple feed-forward neural network, designed to work with N layers and any number of neurons per layer. Is this practically useful in any shape or form? No. Use TensorFlow or PyTorch like a sane person. However, this project stems from a highschool class where we built this from scratch by ourselves in late 2021 (makes me feel old). So in other words, this repo isn't practically useful in any way, but it has been a great way to get more hands on practice with (re)building things from scratch instead of just doing the classic "import X to do Y".

The updated code is written in C++ and was done mostly in 2024, but I am now bringing it to GitHub to properly track changes and to be able to show off something. (This is because my university' academic policies allow for punishing students who upload projects from classes to public repos if their projects are copied by a 3rd party).

*Important*
In the original code, we tested the functionality of the neural network by setting input activations to 0s & 1s and the output activations to a logical combination of input activations. This holdover remains, and current test cases follow the same format.

# Warning about the old code:
I ***highly*** recommend avoiding digging through the OG_Code folder. Please, trust me. It took me a couple weeks of headscratching, (re)diagramming out everything, and questioning my life decisions to decipher the old code I had written during highschool. I geniuenly don't know how the OG code even worked or how it got me an A in the class. Also, there were many quirks to the design and variable naming that were largely due to needing to follow the class's documentation of the mathematical derivation of gradient descent (and of course I lost that documentation). However, my complete lack of remembering that I was working within a class structure is also very apparent. 

tl;dr, my OG code is so bad that only a human could've written it and the best purpose it serves is to prove that poorly documented code quickly goes from "only God and I understand how this works" to "only God knows how the hell this ever worked"
