
python practice
===============

exec vs setattr (time)
----------------------

.. code-block:: python

   import time


   class A:
       def __init__(self):
           self.value = 1
           self.vdict = {}
           self.vdict['0'] = str("self.value")

           start = time.time()
           for i in range(1000):
               setattr(self, self.vdict['0'], 5)
           print('TIME(ms)  is=', 1000 * (time.time() - start))

           start = time.time()
           for i in range(1000):
               exec(self.vdict['0'])
           print('TIME(ms)  is=', 1000 * (time.time() - start))


   if __name__ == '__main__':
       classA = A()
