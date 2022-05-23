#ifndef STD_FUNCTIONS_H
#define STD_FUNCTIONS_H

#include "../driver/heap.h"
#include <initializer_list>
#include <iterator>



namespace stde{


    //CMATH
    int pow(int base, int exponent);
    float pow(float base, float exponent);
    float abs(float value);
    unsigned int abs(int value);
    float sqrt(float value);
    float acos(float x);
    float asin(float x);
    float log(float x);
    float log10(float x);
    float exp(float x);
    float cosh(float x);
    float atan(float value);
    float atan2(float x, float y);
    int max(int x , int y);
    float max(float x, float y);
    float min(float x, float y);
    int min(int x, int y);
    float sinh(float value);
    float cos(float value);
    float sin(float value);

    float fmod(float a, float b);
    float factorial(float n);

    float tan(float value);
    float tanh(float value);

    template<class T>
    T abs(T value){
        return value >= 0 ? value : - value;
    }
    
    template<class T>
    T max(T x, T y){
	    return x > y ? x : y;
    }

    template<class T>
    T min(T x, T y){
	    return x < y ? x : y;
    }

    template<class T>
    T stde::pow(T base, T exponent){
    if( exponent < 0){
        return pow(1/base,stde::abs(exponent) -1);
    }
    return exponent != 0 ? pow(base,exponent-1) : 1;
    };

    template<class T, class E>
    T stde::pow(T base, E exponent){
    if( exponent < 0){
        return pow(1/base,stde::abs(exponent) -1);
    }
    return exponent != 0 ? pow(base,exponent-1) : 1;
    };

    //ALGORITHM
     template<class T>
    T* forward(T* value){
        T* new_value = (T*) malloc(sizeof(value));
        *new_value = *value;
        free(value);
        return new_value;
    }

    template<class InputIt, class UnaryPredicate>
    constexpr InputIt find_if(InputIt first, InputIt last, UnaryPredicate p)
    {
        for (; first != last; ++first) {
            if (p(*first)) {
                return first;
            }
        }
        return last;
    }

    template<class InputIt, class UnaryPredicate>
    constexpr InputIt find_if_not(InputIt first, InputIt last, UnaryPredicate q)
    {
        for (; first != last; ++first) {
            if (!q(*first)) {
                return first;
            }
        }
        return last;
    }

    template< class InputIt, class UnaryPredicate >
    constexpr bool all_of(InputIt first, InputIt last, UnaryPredicate p)
    {
        return stde::find_if_not(first, last, p) == last;
    }

    template< class InputIt, class UnaryPredicate >
    constexpr bool any_of(InputIt first, InputIt last, UnaryPredicate p)
    {
        return stde::find_if(first, last, p) != last;
    }

    template<class ForwardIt>
    ForwardIt max_element(ForwardIt first, ForwardIt last)
    {
    if (first == last) return last;
 
    ForwardIt largest = first;
    ++first;
    for (; first != last; ++first) {
        if (*largest < *first) {
            largest = first;
        }
    }
    return largest;
    }

    template< class T >
    struct plus{
        constexpr T operator()(const T &lhs, const T &rhs) const 
    {
        return lhs + rhs;
    }
    };

    template< class T >
    struct multiplies{
        constexpr T operator()(const T &lhs, const T &rhs) const 
    {
        return lhs * rhs;
    }
    };


template<class ForwardIt>
ForwardIt min_element(ForwardIt first, ForwardIt last)
{
    if (first == last) return last;
 
    ForwardIt smallest = first;
    ++first;
    for (; first != last; ++first) {
        if (*first < *smallest) {
            smallest = first;
        }
    }
    return smallest;
}

template<class InputIt, class T>
constexpr // since C++20
T accumulate(InputIt first, InputIt last, T init)
{
    for (; first != last; ++first) {
        T temp = init;
        init = temp + *first; // std::move since C++20
    }
    return init;
}
template<class InputIt, class T, class BinaryOperation>
constexpr // since C++20
T accumulate(InputIt first, InputIt last, T init, 
             BinaryOperation op)
{
    for (; first != last; ++first) {
        init = op(init, *first); // std::move since C++20
    }
    return init;
}

    //CASSERT

    extern void UnimplementedIrq();



//CASSERT
    void assertt(bool cond);

    //VECTOR
    

template <typename T>
class Vector { //???
private:
    T* arr;
 
    // Variable to store the current capacity
    // of the vector
    unsigned int capacity;
 
    // Variable to store the length of the
    // vector
    unsigned int length;
 
public:
    Vector(unsigned int= 100);
    Vector(std::initializer_list<T> init);
 
    // Function that returns the number of
    // elements in array after pushing the data
    unsigned int push_back(T);
 
    // function that returns the popped element
    T pop_back();
 
    // Function that return the size of vector
    unsigned int size() const;
    T& operator[](unsigned int);

    typedef T* iterator;
    typedef const T* const_iterator;

    iterator begin(){
        return arr;
    }

    iterator end(){
        return arr + length;
    }

    const_iterator begin() const{
        return arr;
    }

    const_iterator end() const {
        return arr + length;
    }

    bool empty(){
        return length == 0;
    }


};

template <typename T>
Vector<T>::Vector(unsigned int n)
    : capacity(100),length(n)
{
    arr = (T*) malloc(sizeof(T)*100);
}

template <typename T>
Vector<T>::Vector(std::initializer_list<T> init){
    arr = (T*) malloc(sizeof(T)*100);
    int counter = 0;
    for(T t : init){
        arr[counter]= t;
        counter++;
    }
    length = counter;// sizeof(arr)/sizeof(arr[0]);
    //stde::assertt(length <= capacity);
}
 
// Template class to insert the element
// in vector
template <typename T>
unsigned int Vector<T>::push_back(T data)
{
    if (length == capacity) {
        T* old = arr;
        //arr = new T[capacity = capacity * 2];
        arr = (T*) malloc(sizeof(T)*capacity*2);
        //stde::memcpy(arr,old,capacity*2);
        arr = old;
        free(old);
    }
    arr[length++] = data;
    return length;
}
 
// Template class to return the popped element
// in vector
template <typename T>
T Vector<T>::pop_back()
{
    return arr[length-- - 1];
}
 
// Template class to return the size of
// vector
template <typename T>
unsigned int Vector<T>::size() const
{
    return length;
}
 
// Template class to return the element of
// vector at given index
template <typename T>
T& Vector<T>::operator[](unsigned int index)
{
    // if given index is greater than the
    // size of vector print Error
    if (index >= length) {
        stde::assertt(index >= length);
    }
 
    // else return value at that index
    return *(arr + index);
}
 
 
// Template class to display element in
// vector
template <typename V>
void display_vector(V& v)
{
    // Declare iterator
    typename V::iterator ptr;
    for (ptr = v.begin(); ptr != v.end(); ptr++) {
        //cout << *ptr << ' ';
    }
}

template<class T, int S>
class array{
    private:
     T* arr;
 
    unsigned int capacity;
 
    unsigned int length;

    public:
    array<T,S>(): length(S), capacity(S) {
        arr =(T*) malloc(sizeof(T) * S);
    }
    array<T,S>(std::initializer_list<T> init);

    unsigned int size();
    T& operator[](unsigned int);


    bool empty();
    int max_size();

    T at(unsigned int i);
    T* data();

    typedef T* iterator;
    typedef const T* const_iterator;

    iterator begin(){
        return arr;
    }

    iterator end(){
        return arr + length ;
    }

    const_iterator begin() const{
        return arr;
    }

    const_iterator end() const {
        return arr + length ;
    }

    
};

template <class T,int S>
array<T,S>::array(std::initializer_list<T> init)
    : capacity(S)
{
    arr = (T*) malloc(sizeof(T)*S);
    int counter = 0;
    for(T t : init){
        arr[counter]= t;
        counter++;
    }
    length = counter;
}

template <typename T,int S>
T& array<T,S>::operator[](unsigned int index)
{
    // if given index is greater than the
    // size of vector print Error
    if (index >= length) {
        assertt(index >= length);
    }
 
    // else return value at that index
    return *(arr + index);
}

template <typename T, int S>
bool array<T,S>::empty(){
    return length == 0;
}


template <typename T, int S>
int array<T,S>::max_size(){
    return capacity;
}

template <typename T, int S>
unsigned int array<T,S>::size(){
    return length;
}

template <typename T, int S>
T array<T,S>::at(unsigned int index){
    return *(arr + index);
}

template <typename T, int S>
T* array<T,S>::data(){
    return *arr;
}



    
    template<bool B, class T = void>
    struct enable_if{};
    
    
    template<class T>
    struct enable_if<true, T> { typedef T type; };


    struct ostream{}; //stub structure in order to allow printing functions work.



template<class T, T v>
struct integral_constant {
    static constexpr T value = v;
    using value_type = T;
    using type = integral_constant; // using injected-class-name
    constexpr operator value_type() const noexcept { return value; }
    constexpr value_type operator()() const noexcept { return value; } // since c++14
};

template< class T > struct remove_cv                   { typedef T type; };
template< class T > struct remove_cv<const T>          { typedef T type; };
template< class T > struct remove_cv<volatile T>       { typedef T type; };
template< class T > struct remove_cv<const volatile T> { typedef T type; };

typedef stde::integral_constant<bool,true> true_type;
typedef stde::integral_constant<bool,false> false_type;

template<typename> struct is_integral_base: stde::false_type {};

template<> struct is_integral_base<bool>: stde::true_type {};
template<> struct is_integral_base<int>: stde::true_type {};
template<> struct is_integral_base<short>: stde::true_type {};
template<> struct is_integral_base<long int>: stde::true_type {};
template<> struct is_integral_base<long long int>: stde::true_type {};
template<> struct is_integral_base<unsigned int>: stde::true_type {};
template<> struct is_integral_base<long unsigned int>: stde::true_type {};
template<> struct is_integral_base<long long unsigned int>: stde::true_type {};

template<typename T> struct is_integral: is_integral_base<typename stde::remove_cv<T>::type> {};
/*
template< class T >
struct is_integral
{
    static const bool value ;
    typedef stde::integral_constant<bool, value> type;
};*/




    template< class T > struct remove_const                { typedef T type; };
    template< class T > struct remove_const<const T>       { typedef T type; };

    struct input_iterator_tag { }; 	
    struct output_iterator_tag { };
	
    struct forward_iterator_tag : public input_iterator_tag { };

    
    /*
    template< class C >
    auto begin( C& c ) -> decltype(c.begin());

    template< class C >
    auto end( C& c ) -> decltype(c.end());
    */



   template<class T>
   auto begin(T c){
       return c.begin;
   }

   template<class T>
   auto end(T c) {
       return c.end;
   }


//#include <iostream>
//using namespace std;
  
// Custom Map Class
class Map {
private:
    Map* iterator(int first)
    {
        // A temporary variable created 
        // so that we do not
        // loose the "root" of the tree
        Map* temp = root;
  
        // Stop only when either the key is found 
        // or we have gone further the leaf node
        while (temp != nullptr && 
               temp->first != first) {
  
            // Go to left if key is less than 
            // the key of the traversed node
            if (first < temp->first) {
                temp = temp->left;
            }
  
            // Go to right otherwise
            else {
                temp = temp->right;
            }
        }
        // If there doesn't exist any element 
        // with first as key, nullptr is returned
        return temp;
    }
  
    // Returns the pointer to element 
    // whose key matches first.
    // Specially created for search method
    // (because search() is const qualified).
    const Map* iterator(int first) const
    {
        Map* temp = root;
        while (temp != nullptr 
               && temp->first != first) {
            if (first < temp->first) {
                temp = temp->left;
            }
            else {
                temp = temp->right;
            }
        }
        return temp;
    }
  
    // The const property is used to keep the
    // method compatible with the method "const
    // int&[]operator(int) const" 
    // Since we are not allowed to change 
    // the class attributes in the method 
    // "const int&[]operator(int) const" 
    // we have to assure the compiler that 
    // method called(i.e "search") inside it
    // doesn't change the attributes of class
    const int search(int first) const
    {
        const Map* temp = iterator(first);
        if (temp != nullptr) {
            return temp->second;
        }
        return 0;
    }
    
   
    // Utiliity function to return the Map* object
    // with its members initilized 
    // to default values except the key
    Map* create(int first)
    {
        Map* newnode = (Map*) malloc(sizeof(Map));
        newnode->first = first;
        newnode->second = 0;
        newnode->left = nullptr;
        newnode->right = nullptr;
        newnode->par = nullptr;
  
        // Depth of a newnode shall be 1 
        // and not zero to differentiate 
        // between no child (which returns
        // nullptr) and having child(returns 1)
        newnode->depth = 1;
        return newnode;
    }
  
    // All the rotation operation are performed 
    // about the node itself
    // Performs all the linking done when there is
    // clockwise rotation performed at node "x"
    void right_rotation(Map* x)
    {
        Map* y = x->left;
        x->left = y->right;
        if (y->right != nullptr) {
            y->right->par = x;
        }
        if (x->par != nullptr && x->par->right == x) {
            x->par->right = y;
        }
        else if (x->par != nullptr && x->par->left == x) {
            x->par->left = y;
        }
        y->par = x->par;
        y->right = x;
        x->par = y;
    }
  
    // Performs all the linking done when there is
    // anti-clockwise rotation performed at node "x"
    void left_rotation(Map* x)
    {
        Map* y = x->right;
        x->right = y->left;
        if (y->left != nullptr) {
            y->left->par = x;
        }
        if (x->par != nullptr && x->par->left == x) {
            x->par->left = y;
        }
        else if (x->par != nullptr && x->par->right == x) {
            x->par->right = y;
        }
        y->par = x->par;
        y->left = x;
        x->par = y;
    }
  
    // Draw the initial and final graph of each
    // case(take case where every node has two child) 
    // and update the nodes depth before any rotation
  
    void helper(Map* node)
    {
        // If left skewed
        if (depthf(node->left) 
            - depthf(node->right) > 1) {
  
            // If "depth" of left subtree of 
            // left child of "node" is 
            // greater than right
            // subtree of left child of "node"
            if (depthf(node->left->left)
                > depthf(node->left->right)) {
                node->depth
                    = stde::max(depthf(node->right) + 1,
                          depthf(node->left->right) + 1);
                node->left->depth
                    = stde::max(depthf(node->left->left) + 1,
                          depthf(node) + 1);
                right_rotation(node);
            }
  
            // If "depth" of right subtree 
            // of left child of "node" is 
            // greater than 
            // left subtree of left child
            else {
                node->left->depth = stde::max(
                    depthf(node->left->left) + 1,
                    depthf(node->left->right->left) 
                    + 1);
                node->depth 
                    = stde::max(depthf(node->right) + 1,
                      depthf(node->left->right->right) + 1);
                node->left->right->depth
                    = stde::max(depthf(node) + 1,
                          depthf(node->left) + 1);
                left_rotation(node->left);
                right_rotation(node);
            }
        }
  
        // If right skewed
        else if (depthf(node->left) 
                 - depthf(node->right) < -1) {
  
            // If "depth" of right subtree of right
            // child of "node" is greater than 
            // left subtree of right child
            if (depthf(node->right->right)
                > depthf(node->right->left)) {
                node->depth
                    = stde::max(depthf(node->left) + 1,
                          depthf(node->right->left) + 1);
                node->right->depth
                    = stde::max(depthf(node->right->right) + 1,
                          depthf(node) + 1);
                left_rotation(node);
            }
  
            // If "depth" of left subtree 
            // of right child of "node" is 
            // greater than that of right
            // subtree of right child of "node"
            else {
                node->right->depth = stde::max(
                    depthf(node->right->right) + 1,
                    depthf(node->right->left->right) + 1);
                node->depth = stde::max(
                    depthf(node->left) + 1,
                    depthf(node->right->left->left) + 1);
                node->right->left->depth
                    = stde::max(depthf(node) + 1,
                          depthf(node->right) + 1);
                right_rotation(node->right);
                left_rotation(node);
            }
        }
    }
  
    // Balancing the tree about the "node"
    void balance(Map* node)
    {
        while (node != root) {
            int d = node->depth;
            node = node->par;
            if (node->depth < d + 1) {
                node->depth = d + 1;
            }
            if (node == root
                && depthf(node->left) > 1) {
                if (depthf(node->left->left)
                    > depthf(node->left->right)) {
                    root = node->left;
                }
                else {
                    root = node->left->right;
                }
                helper(node);
                break;
            }
            else if (node == root
                     && depthf(node->left)
                                - depthf(node->right)
                            < -1) {
                if (depthf(node->right->right)
                    > depthf(node->right->left)) {
                    root = node->right;
                }
                else {
                    root = node->right->left;
                }
                helper(node);
                break;
            }
            helper(node);
        }
    }
  
    // Utility method to return the 
    // "depth" of the subtree at the "node"
  
    int depthf(Map* node)
    {
        if (node == nullptr)
  
            // If it is null node
            return 0;
        return node->depth;
    }
  
    Map* insert(int first)
    {
        cnt++;
        Map* newnode = create(first);
        if (root == nullptr) {
            root = newnode;
            return root;
        }
        Map *temp = root, *prev = nullptr;
        while (temp != nullptr) {
            prev = temp;
            if (first < temp->first) {
                temp = temp->left;
            }
            else if (first > temp->first) {
                temp = temp->right;
            }
            else {
                free(newnode);
                cnt--;
                return temp;
            }
        }
        if (first < prev->first) {
            prev->left = newnode;
        }
        else {
            prev->right = newnode;
        }
        newnode->par = prev;
        balance(newnode);
        return newnode;
    }
  
    // Returns the previous node in 
    // inorder traversal of the AVL Tree.
    Map* inorderPredecessor(Map* head)
    {
        if (head == 0)
            return head;
        while (head->right != 0) {
            head = head->right;
        }
        return head;
    }
  
    // Returns the next node in 
    // inorder traversal of the AVL Tree.
    Map* inorderSuccessor(Map* head)
    {
        if (head == 0)
            return head;
        while (head->left != 0) {
            head = head->left;
        }
        return head;
    }
  
public:
    // Root" is kept static because it's a class
    // property and not an instance property
    static class Map* root;
    static int cnt;
  
    // "first" is key and "second" is value
    Map *left, *right, *par;
    int first, second, depth;
  
    // #overloaded [] operator for assignment or
    // inserting a key-value pairs in the map 
    // since it might change the members of 
    // the class therefore this is
    // invoked when any assignment is done
    int& operator[](int key) { 
        return insert(key)->second; 
    }
  
    // #Since we have two methods with 
    // the same name "[]operator(int)" and 
    // methods/functions cannot be
    // distinguished by their return types 
    // it is mandatory to include a const 
    // qualifier at the end of any of the methods
  
    // This method will be called from a const 
    // reference to the object of Map class
  
    // It will not be called for assignment 
    // because it doesn't allow to change 
    // member variables
  
    // We cannot make it return by reference 
    // because the variable "temp" returned 
    // by the "search" method is
    // statically allocated and therefore 
    // it's been destroyed when it is called out
    const int operator[](int key) const
    {
        return search(key);
    }
  
    // Count returns whether an element 
    // exists in the Map or not
    int count(int first)
    {
        Map* temp = iterator(first);
        if (temp != nullptr) {
            return 1;
        }
        return 0;
    }
  
    // Returns number of elements in the map
    int size(void) { 
        return cnt; 
    }
  
    // Removes an element given its key
    void erase(int first, Map* temp = root)
    {
        Map* prev = 0;
        cnt--;
        while (temp != 0 && 
               temp->first != first) {
            prev = temp;
            if (first < temp->first) {
                temp = temp->left;
            }
            else if (first > temp->first) {
                temp = temp->right;
            }
        }
        if (temp == nullptr) {
            cnt++;
            return;
        }
        if (cnt == 0 && temp == root) {
            free(temp);
            root = nullptr;
            return;
        }
        Map* l 
            = inorderPredecessor(temp->left);
        Map* r 
            = inorderSuccessor(temp->right);
        if (l == 0 && r == 0) {
            if (prev == 0) {
                root = 0;
            }
            else {
                if (prev->left == temp) {
                    prev->left = 0;
                }
                else {
                    prev->right = 0;
                }
                free(temp);
                balance(prev);
            }
            return;
        }
        Map* start;
        if (l != 0) {
            if (l == temp->left) {
                l->right = temp->right;
                if (l->right != 0) {
                    l->right->par = l;
                }
                start = l;
            }
            else {
                if (l->left != 0) {
                    l->left->par = l->par;
                }
                start = l->par;
                l->par->right = l->left;
                l->right = temp->right;
                l->par = 0;
                if (l->right != 0) {
                    l->right->par = l;
                }
                l->left = temp->left;
                temp->left->par = l;
            }
            if (prev == 0) {
                root = l;
            }
            else {
                if (prev->left == temp) {
                    prev->left = l;
                    l->par = prev;
                }
                else {
                    prev->right = l;
                    l->par = prev;
                }
                free(temp);
            }
            balance(start);
            return;
        }
        else {
            if (r == temp->right) {
                r->left = temp->left;
                if (r->left != 0) {
                    r->left->par = r;
                }
                start = r;
            }
            else {
                if (r->right != 0) {
                    r->right->par = r->par;
                }
                start = r->par;
                r->par->left = r->right;
                r->left = temp->left;
                r->par = 0;
                if (r->left != 0) {
                    r->left->par = r;
                }
                r->right = temp->right;
                temp->right->par = r;
            }
            if (prev == 0) {
                root = r;
            }
            else {
                if (prev->right == temp) {
                    prev->right = r;
                    r->par = prev;
                }
                else {
                    prev->left = r;
                    r->par = prev;
                }
                free(temp);
            }
            balance(start);
            return;
        }
    }
    
    // Returns if the map is empty or not
    bool empty(void)
    {
        if (root == 0)
            return true;
        return false;
    }
    
    // Given the key of an element it updates 
    // the value of the key
    void update(int first, int second)
    {
        Map* temp = iterator(first);
        if (temp != nullptr) {
            temp->second = second;
        }
    }
  
    // Recursively calling itself and 
    // deleting the root of
    // the tree each time until the map 
    // is not empty
    void clear(void)
    {
        while (root != nullptr) {
            erase(root->first);
        }
    }
  
    // Inorder traversal of the AVL tree
    void iterate(Map* head = root)
    {
        if (root == 0)
            return;
        if (head->left != 0) {
            iterate(head->left);
        }
        //cout << head->first << ' ';
        if (head->right != 0) {
            iterate(head->right);
        }
    }
  
    // Returns a pointer/iterator to the element 
    // whose key is first
    Map* find(int first) { 
        return iterator(first); 
    }
  
    // Overloaded insert method, 
    // takes two parameters - key and value
    void insert(int first, int second)
    {
        Map* temp = iterator(first);
        if (temp == nullptr) {
            insert(first)->second = second;
        }
        else {
            temp->second = second;
        }
    }
};

    inline void* memcpy(void*dest, void*source,int size){
        char* c_source = (char*) source;
        char* c_dest = (char*) dest;
        for(int i = 0;i < size; i++){
            c_dest[i] = c_source[i];
        }
        return c_dest;
    }  
 
}

    
#endif

