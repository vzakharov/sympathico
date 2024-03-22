# SymPathiCo: Harnessing Evolutionary Adaptation for Calculus-Free Neural Computing

## Abstract

In the rapidly evolving landscape of machine learning, traditional neural network architectures have consistently relied on complex calculus and weight-based adjustments for training and optimization. Despite their successes, these networks face scalability challenges, veer more and more from the intuitive natural analogies that inspired them, and struggle with the inherent complexities of high-dimensional data spaces that exacerbate the curse of dimensionality.

This paper introduces SymPathiCo (Symbolic Pathing Colonies), a novel approach that eschews conventional weight matrices, backpropagation, and even layered architectures in favor of a dynamic, arbitrarily wired, evolutionary framework. By conceptualizing the neural network as a colony of symbolic paths—each representing a possible sequential activation of neurons—SymPathiCo aligns more closely with organically inspired principles seen in natural processes such as waterways and ant trails, where paths, not abstract matrices, underpin essential processes.

SymPathiCo's evolutionary adaptation of paths within a neural colony, driven by real-time interaction with input-output pairs, proposes a novel method of “training” through the replication of successful activation sequences. The paper also introduces a necessity-driven activation function based on a “spillover” concept, which models non-linearity, essential for complex problem-solving, in a calculus-free and intuitively natural manner.

Given the network’s structure as a vast population of paths, SymPathiCo hints at unprecedented scalability, with the capacity for thousands of machines to asynchronously contribute to the evolution of the colony. This intuition, while yet to be rigorously tested, suggests a new frontier in neural network architectures’ scalability and distributed computing capabilities.

To facilitate exploration and experimentation with SymPathiCo, we provide a software framework including a Python package and a Colab notebook, enabling the community to test and extend the approach further.

## How Watersheds Evolve

Before we delve into the mathematics of our approach, let's first immerse ourselves in the intuitive landscape that inspires it. Imagine standing at the crest of a vast, untouched terrain, where the potential for life-giving waterways exists in every undulation and declivity. This is the genesis of "Watersheds," a concept that reimagines neural networks not as static constructs of nodes and edges but as ever-evolving waterways, carving their paths through the landscape of information.

### Laying Out the Landscape

In the realm of Watersheds, each neuron is a junction within an intricate network of rivers and streams, poised to guide the flow of information from source to sea. Our task, as architects of this system, is to map out the terrain—defining the bounds of our landscape, where each junction is marked by a unique identifier, a number that signifies its place in the flow of data.

### Carving the Pathways

With the terrain laid out, our landscape awaits the nourishing rains—the input data. As these first droplets gather at the highest junctions, they begin their descent, guided by the topology of our design. Their journey is one of exploration and discovery, seeking paths that lead them efficiently to their destinations.

* Routing the Droplets: In Watersheds, each droplet is a bearer of information, navigating the network with a single mandate: to flow downwards. This directed movement, from higher-numbered junctions to lower, ensures a unidirectional flow that mirrors the cascading nature of waterfalls, each drop moving inexorably towards the ocean depths—the output neurons.
    
* From Rain to Rivers: As droplets traverse junctions, they encounter decision points—moments to continue along established pathways or forge new ones. These decisions, while seemingly random, are guided by the inherent logic of the landscape and the destination of the droplets. Successful paths, those that facilitate a seamless journey from source to sea, are earmarked for replication, echoing the way trails are beaten through forests by repeated use.
    

### The Saturation Threshold

Central to the concept of Watersheds is the notion of saturation—a junction's capacity to guide droplets without becoming overwhelmed. This mechanism acts as a natural regulator, ensuring no single path becomes a floodway. It is through this balance that the landscape maintains its diversity and resilience, allowing for the redistribution of flows when certain paths become untenable.

### Replicating Success

The completion of a droplet's journey from its source to the desired sea level is a cause for celebration and replication. This natural selection process, akin to reinforcing the banks of a river, ensures that future flows are more likely to follow these proven paths, strengthening the network's overall efficiency and adaptability.

### The Evolution of Pathways

Over time, as more rains fall and more paths are etched into the landscape, Watersheds transforms into a vibrant tapestry of flowing information. This dynamic, self-organizing system, guided by the principles of evolution and adaptation, ensures that only the most efficient pathways prevail, while those less traveled gradually fade, reclaimed by the natural order.

Watersheds invites us to envision neural networks as living, breathing ecosystems, where information flows like water, seeking efficiency and harmony in its passage. It is a journey through a landscape that learns and adapts, not through the rigid application of rules, but through the organic process of growth, selection, and renewal.