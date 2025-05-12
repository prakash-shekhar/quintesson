# quintesson
quintesson makes fine-tuning and memory-editing in Language Models accessible to people with a non-technical background. Now, anybody can train and host their own Language Models without ever having to open a terminal or code a single line of code. 

Currently, there are similar solutions with custom GPTs and RAG.  But, Parameter-Efficient Fine Tuning changes the model’s behavior internally and persistently — not just fetching knowledge and MemIT edits factual knowledge or associations directly in model weights, rather than relying on external documents. RAG is great for flexible, fresh knowledge but depends on retrieval quality and external document accuracy. Whereas, PEFT + MemIT = persistent, low-latency, truly integrated behavioral or factual change.

If RAG is like giving the model a better library, PEFT and MemIT are like neurosurgery. This app opens that power to everyone, without them needing to understand tensors or GPUs.

Here are two example use cases:

User: Sarah, a licensed therapist with no coding background
Goal: Chatbot that supports her clients between sessions, using her specific therapeutic approach (CBT + mindfulness), language, and tone.
Use:
- Uploads anonymized client notes and therapy session scripts.
- Uses a no-code interface to fine-tune the model with PEFT to match her vocabulary, techniques, and prompts.
- Uses MemIT to inject key psychological facts or frameworks (e.g., “CBT involves identifying and challenging cognitive distortions”).
- Why not CustomGPT/RAG?: She would have to send her confidential client data to a third-party whereas with Quintesson she owns her data.

User: James, founder of a small handmade furniture store
Goal: Automate customer support with a model that knows his product line, return policies, and brand tone (friendly, handcrafted).
Use:
- Feeds in past support emails and policy docs to fine-tune the model with PEFT for handling typical questions.
- Uses MemIT to edit specific facts like return periods, product specs, and pricing logic.
- Instantly tests and iterates through a GUI to refine how the model responds to edge cases like damage claims or delivery delays.
- Why not RAG?: He doesn’t want to maintain or troubleshoot a document store and vector DB. He needs fast, baked-in answers with low hosting costs and no infra maintenance