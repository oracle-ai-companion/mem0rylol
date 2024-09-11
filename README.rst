===============================================
mem0rylol: The Sophisticated AI Memory Layer
===============================================

.. image:: https://img.shields.io/badge/version-0.2.0-blue.svg
   :target: https://pypi.org/project/mem0rylol/
   :alt: Version

.. image:: https://img.shields.io/badge/license-GNU-green.svg
   :target: https://www.gnu.org/licenses/gpl-3.0.en.html
   :alt: License

.. image:: https://img.shields.io/badge/python-3.10%2B-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python version

mem0rylol is a cutting-edge AI memory layer designed to enhance the capabilities of AI systems by providing sophisticated memory management and retrieval.

Features
--------

- Advanced memory storage and retrieval
- Seamless integration with LangChain
- Support for multiple AI models
- Efficient data handling with LanceDB

Installation
------------

Install mem0rylol using pip:

.. code-block:: bash

   pip install mem0rylol

For detailed installation instructions, please refer to the `installation guide <docs/installation.rst>`_.

Quick Start
-----------

Here's a simple example to get you started:

.. code-block:: python

   from mem0rylol import MemoryManager
   from mem0rylol.memory.memory_types import Memory
   from mem0rylol.schemas.lancedb import LanceDBSchema

   # Initialize the memory manager
   memory_manager = MemoryManager(table_name="my_memories", schema_cls=LanceDBSchema)

   # Extract and store memories
   new_memories = memory_manager.extract_memories("John is a software engineer living in New York who loves pizza.", {})
   for memory in new_memories:
       memory_manager.add_memory(memory)

   # Retrieve and update memories
   similar_memories = memory_manager.similarity_search("John's food preferences")
   updated_memories = memory_manager.update_memories(Memory(text="John now prefers pasta"), similar_memories)

   # Generate a response based on memories
   response = memory_manager.generate_response("What do we know about John?")
   print(response)

Documentation
-------------

For comprehensive documentation, including API references and usage examples, please visit our `documentation <docs/index.rst>`_.

Contributing
------------

We welcome contributions! Please see our `contributing guidelines <CONTRIBUTING.rst>`_ for more information on how to get involved.

License
-------

mem0rylol is released under the GNU General Public License v3.0. See the `LICENSE <LICENSE.txt>`_ file for more details.

.. code-block:: text
   :name: license-snippet

   startLine: 1
   endLine: 30

Contact
-------

For questions, suggestions, or support, please contact the project maintainer:

- **Author**: toeknee
- **Email**: [Your contact email]
- **GitHub**: [Your GitHub profile]

Acknowledgments
---------------

We would like to thank the following projects and libraries that make mem0rylol possible:

- LangChain
- LanceDB
- Pydantic

Stay Connected
--------------

- Follow us on Twitter: [@mem0rylol]
- Join our Discord community: [Discord invite link]
- Subscribe to our newsletter: [Newsletter signup link]

Happy coding with mem0rylol!