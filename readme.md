# SymPathiCo: Harnessing Evolutionary Adaptation for Calculus-Free Neural Computing

## Abstract

In the rapidly evolving landscape of machine learning, traditional neural network architectures have consistently relied on complex calculus and weight-based adjustments for training and optimization. Despite their successes, these networks face scalability challenges, veer more and more from the intuitive natural analogies that inspired them, and struggle with the inherent complexities of high-dimensional data spaces that exacerbate the curse of dimensionality.

This paper introduces SymPathiCo (Symbolic Pathing Colonies), a novel approach that eschews conventional weight matrices, backpropagation, and even layered architectures in favor of a dynamic, arbitrarily wired, evolutionary framework. By conceptualizing the neural network as a colony of symbolic paths—each representing a possible sequential activation of neurons—SymPathiCo aligns more closely with organically inspired principles seen in natural processes such as waterways and ant trails, where paths, not abstract matrices, underpin essential processes.

SymPathiCo's evolutionary adaptation of paths within a neural colony, driven by real-time interaction with input-output pairs, proposes a novel method of “training” through the replication of successful activation sequences. The paper also introduces a necessity-driven activation function based on a “spillover” concept, which models non-linearity, essential for complex problem-solving, in a calculus-free and intuitively natural manner.

Given the network’s structure as a vast population of paths, SymPathiCo hints at unprecedented scalability, with the capacity for thousands of machines to asynchronously contribute to the evolution of the colony. This intuition, while yet to be rigorously tested, suggests a new frontier in neural network architectures’ scalability and distributed computing capabilities.

To facilitate exploration and experimentation with SymPathiCo, we provide a software framework including a Python package and a Colab notebook, enabling the community to test and extend the approach further.

## Training SymPathiCo: Navigating the Neural Waterways

Within the framework of SymPathiCo, we conceive a neural network not as a static structure of weights and biases but as a living landscape of waterways. Here, each neuron is a junction in a vast network of rivers and streams, and the information flows like water through these junctions.

### Laying Out the Landscape

Before the first drop falls, we chart out our landscape. It's a realm where each junction (neuron) is defined by a unique number, set against a backdrop of predefined bounds. This terrain is meticulously mapped to ensure a directed flow, akin to designing a water park where every slide and channel has its purpose, guiding water from the entrance (input neurons) to the exit (output neurons).

### Carving the Pathways

As our network begins, it's a dry expanse, waiting for the life-giving flow of information. This changes with the introduction of our first rainstorm, the input data. Droplets gather at the topmost junctions, ready to embark on their journey through the network.

* Routing the Droplets: Each droplet represents a unit of information, seeking the path of least resistance towards its destination. However, it's bound by the natural law of this landscape: it can only flow downwards, from one junction to the next, mirroring the increasing order of our numbered neurons. This ensures that while our rivers can be wide and winding, they do not loop or backtrack, maintaining a forward momentum towards the outputs.

* From Rain to Rivers: When a droplet encounters a junction, it has a choice to continue along existing pathways or, if none are suitable or existing, to carve new ones. The journey is random, exploring new connections between junctions, always aiming to reach the sea level—the output neurons. If a path proves successful, leading a droplet from its source to the desired destination, it is marked for replication, reinforcing this route within the landscape.

### The Saturation Threshold

A unique aspect of our waterway system is the concept of saturation at each junction. Each can handle only so many droplets before reaching a threshold, beyond which it can no longer pass water efficiently. This natural regulation mechanism prevents flooding and ensures that our landscape remains balanced, with no single path becoming overwhelmingly dominant. When a junction overflows, it's a sign to explore alternative routes, mimicking the dynamic adaptation seen in natural systems.

### Replicating Success

Upon completing a journey, if the droplets have successfully navigated from source to sea following the desired path, we replicate this path. It's akin to reinforcing a trail in the wilderness, making it more likely for future travelers (or in our case, information) to follow. This process of selection and replication allows the network to evolve naturally, strengthening successful paths over time.

### The Evolution of Pathways

As more rain falls and more paths are carved, our landscape becomes a complex network of flowing information. Each successful journey reinforces the pathways, making them more prominent features of the landscape. Unsuccessful or unused paths gradually fade, reclaimed by the terrain. This dynamic, evolving system mirrors the process of natural selection, where only the most efficient and effective paths survive and thrive.

In essence, SymPathiCo's approach to neural network training is a journey through a constantly evolving landscape, where information flows like water, seeking the most efficient paths through a network of interconnected junctions. It's a system that learns and adapts, not through rigid calculations and adjustments, but through the natural, intuitive process of evolution and adaptation.