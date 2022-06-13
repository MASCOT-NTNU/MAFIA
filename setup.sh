g++ -c -fPIC AH3D2.cpp -o AH3D2.o
g++ -shared -W1,libAH3D2.so -o libAH3D2.so AH3D2.o
echo "SET UP SUCCESFULLY!"
