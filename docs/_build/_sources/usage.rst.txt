Usage
=====

Here's a basic example of how to use mem0rylol:

.. code-block:: python

   from mem0rylol import MemoryManager

   memory_manager = MemoryManager(table_name="my_table", schema_cls=MySchema)
   memory_manager.add_memory(Memory(text="Hello, world!"))
   results = memory_manager.similarity_search("Hello")