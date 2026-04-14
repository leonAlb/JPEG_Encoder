#ifndef JPEG_PROJEKT_HUFFMAN_H
#define JPEG_PROJEKT_HUFFMAN_H

#include <iostream>
#include <cstdint>
#include <vector>
#include <optional>
#include <queue>
#include <iomanip>
#include <bitset>
#include <BitStream.h>

using namespace std;

struct Trunk {
    Trunk *prev;
    string str;

    Trunk(Trunk *prev, const string &str) : prev(prev), str(str) {
    }
};


template<typename T>
class Node {
public:
    uint32_t data = 0;
    int level = 0;
    int frequency;
    optional<T> symbol = nullopt;
    Node *nodeLeft = nullptr;
    Node *nodeRight = nullptr;

    // Leave
    explicit Node(T t, const int frequency)
        : frequency(frequency),
          symbol(t),
          nodeLeft(nullptr), nodeRight(nullptr) {
    }

    // Useless node
    explicit Node(nullopt_t)
        : frequency(-1),
          symbol(nullopt),
          nodeLeft(nullptr), nodeRight(nullptr) {
    }

    // Internal parent node
    Node(const int frequency, const int level, Node *left, Node *right)
        : level(level),
          frequency(frequency),
          nodeLeft(left), nodeRight(right) {
    }

    // Default (root)
    Node() : frequency(0), symbol(nullopt), nodeLeft(nullptr), nodeRight(nullptr) {
    }
};

template<typename T>
struct compareFrequency {
    bool operator()(const Node<T> *l, const Node<T> *r) const {
        return l->frequency > r->frequency;
    }
};

template<typename T>
struct compareLevel {
    bool operator()(const Node<T> *l, const Node<T> *r) const {
        return l->level < r->level;
    }
};

template<typename T>
struct compareLevelAndSymbol {
    bool operator()(const Node<T> *l, const Node<T> *r) const {
        // First level
        if (l->level != r->level)
            return l->level < r->level;

        // Then symbol
        if (l->symbol && r->symbol)
            return l->symbol.value() < r->symbol.value();

        // Leaves first
        if (l->symbol) return true;
        if (r->symbol) return false;

        return false;
    }
};


template<typename T>
class Builder {
public:
    Node<T> *root = nullptr;
    unordered_map<T, int> frequencyMap;
    vector<vector<Node<T> *> > nodesList;
    unordered_map<T, pair<uint32_t,int>> codeMap;

    int levels = 0;

    // Clean up memory
    ~Builder() {
        deleteTree(root);
    }

    void deleteTree(Node<T> *node) {
        if (!node) return;
        deleteTree(node->nodeLeft);
        deleteTree(node->nodeRight);
        delete node;
    }

    void set_frequency_map(unordered_map<T, int> map) {
        frequencyMap = map;
    }

    void getFreqFromChannel(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &channel) {
        for (int y = 0; y < channel.rows(); ++y) {
            for (int x = 0; x < channel.cols(); ++x) {
                ++frequencyMap[channel(y, x)];
            }
        }
    }

    void buildTree() {
        if (frequencyMap.empty()) return;

        priority_queue<Node<T> *, vector<Node<T> *>, compareFrequency<T> > pq;

        // Creates leave for each entry
        for (const auto &[symbol, frequency]: frequencyMap) {
            pq.push(new Node<T>(symbol, frequency));
        }

        // Use all nodes until only one left (root)
        while (pq.size() > 1) {
            Node<T> *left = pq.top();
            pq.pop();
            Node<T> *right = pq.top();
            pq.pop();

            // Frequency and level of new parent node
            int sumFrequency = left->frequency + right->frequency;
            int sumLevel = max(left->level, right->level) + 1;

            auto *parent = new Node<T>(sumFrequency, sumLevel, left, right);

            pq.push(parent);
        }

        root = pq.top();
        pq.pop();
    }

    // Breadth search
    void createBinaryTreeAs2DVector() {
        if (root == nullptr) return;

        assignHuffmanCodes(root);

        queue<Node<T> *> queue;
        queue.push(root);
        int currentLevel = 0;

        while (!queue.empty()) {
            const int size = queue.size();
            nodesList.push_back({});
            for (int i = 0; i < size; i++) {
                Node<T> *currentNode = queue.front();
                queue.pop();
                currentNode->level = currentLevel;

                nodesList[currentLevel].push_back(currentNode);

                if (currentNode->nodeLeft) queue.push(currentNode->nodeLeft);
                if (currentNode->nodeRight) queue.push(currentNode->nodeRight);
            }
            currentLevel++;
        }
        levels = nodesList.size();
    }

    void assignHuffmanCodes(Node<T>* node, uint32_t code = 0, int length = 0) {
        if (!node) return;

            node->data = code;
            node->level = length;

        if (node->nodeLeft)
            assignHuffmanCodes(node->nodeLeft, code << 1, length + 1);
        if (node->nodeRight)
            assignHuffmanCodes(node->nodeRight, (code << 1) | 1, length + 1);
    }

    void buildDepthLimitedTree(int limit)
        {
            if (nodesList.size() <= limit)
                return;

            int current_list = nodesList.size()-1;

            while (current_list > limit)
            {
                vector<Node<T>*> leaves;
                for (Node<T>* node : nodesList[current_list])
                {
                    if (node->symbol.has_value())
                        leaves.push_back(node);
                }

                if (leaves.size() % 2 != 0)
                    throw std::runtime_error("ERROR: ODD NUMBER OF LEAVES");

                for (int i = 0; i < leaves.size(); i += 2)
                {
                    Node<T>* x = leaves[i];
                    Node<T>* y = leaves[i + 1];
                    replaceParentWithX(x, y);
                    findNextZ(current_list, y);
                }
                nodesList.pop_back();
                current_list--;
                levels--;
            }
        }

    void findNextZ(int currentLevel, Node<T>* y) {
        if (currentLevel < 2)
            throw std::out_of_range("FEHLER: currentLevel < 2!");

        for (int targetLevel = currentLevel - 2; targetLevel >= 0; targetLevel--) // Search 2 levels deep
        {
            auto& vec = nodesList[targetLevel];
            for (auto it = vec.begin(); it != vec.end(); ++it)
            {
                Node<T>* original = *it;
                if (original && original->symbol.has_value()) {
                    Node<T>* z = new Node<T>(*original);
                    original->symbol = std::nullopt;
                    original->nodeRight = y;
                    original->nodeLeft = z;

                    // create new codes
                    y->data = (z->data << 1) | 1;
                    y->level = z->level + 1;
                    z->data <<= 1;
                    z->level++;

                    nodesList[y->level].push_back(y);
                    nodesList[z->level].push_back(z);
                    return;
                }
            }
        }
        throw std::out_of_range("ERROR: No usable Z!");
    }


    void replaceParentWithX(Node<T>* x, Node<T>* y)
    {
        int parentLevel = x->level - 1;
        if (parentLevel < 0 || parentLevel >= nodesList.size())
            throw std::runtime_error("ERROR: parentLevel out of range!");

        auto& vec = nodesList[parentLevel];
        for (Node<T>* parent : vec)
        {
            if (!parent || !parent->nodeLeft)
                continue;

            if (parent->nodeLeft->symbol == x->symbol && parent->nodeRight->symbol == y->symbol || parent->data == (x->data >> 1))
            {
                // Parent takes over symbol and frequency of x
                parent->symbol = x->symbol;
                parent->frequency = x->frequency;

                delete x;

                // new X (leave)
                parent->nodeLeft  = nullptr;
                parent->nodeRight = nullptr;

                return;
            }
        }
        throw std::runtime_error("ERROR: No parent node found!");
    }


    void buildRightGrowingTreeFrom2DVector()
    {
        levels = nodesList.size();
        for (size_t i = 0; i < levels - 1; ++i) {
            auto parents = nodesList[i];
            auto children = nodesList[i + 1];

            sort(parents.begin(), parents.end(), compareLevelAndSymbol<T>());
            sort(children.begin(), children.end(), compareLevelAndSymbol<T>());

            while (!children.empty()) {
                int frequencySum = 0;
                // Takes node from right side back
                auto parent = parents.back();
                parents.pop_back();

                // Adds Node
                auto attachChild = [&](Node<T> *&targetNode, bool isRight) {
                    auto child = children.back();
                    children.pop_back();
                    if (child != nullptr)
                    {
                        // Set Code
                        child->data = (parent->data << 1) | (isRight ? 1 : 0);
                        child->level = i + 1;
                        targetNode = child;
                        frequencySum += child->frequency;

                        if (child->symbol.has_value())
                        {
                            codeMap[child->symbol.value()] = {child->data, child->level};
                        }
                    }
                };
                attachChild(parent->nodeRight, true);
                attachChild(parent->nodeLeft, false);
                parent->frequency = frequencySum;
            }
        }
        preventOneStarCode(); // Workaround: avoid a rightmost leaf with an all-1 code.
    }

    void printCodes(const Node<T> *currentNode, int currentLevel) {
        if (currentNode == nullptr) {
            return;
        }
        if (!currentNode->nodeLeft && !currentNode->nodeRight && currentNode->symbol.has_value()) {
            cout << "Code for " << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(currentNode->symbol.value()) << ": " << std::dec;
            for (int i = currentLevel - 1; i >= 0; i--) {
                uint32_t tmp = currentNode->data;
                tmp = tmp >> i;
                cout << (tmp & 0x1);
            }
            cout << " (" << currentNode->data << ")" << std::endl;
            return;
        }
        if (currentLevel == levels)
            return;

        printCodes(currentNode->nodeLeft, currentLevel + 1);
        printCodes(currentNode->nodeRight, currentLevel + 1);
    }

    void encodePixelsUint(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &channel, BitStream &bitStream) {
        for (int y = 0; y < channel.rows(); ++y) {
            for (int x = 0; x < channel.cols(); ++x) {
                auto val = static_cast<uint32_t>(channel(y, x));
                auto [bits, bitCount] = codeMap[val];
                bitStream.writeBitsBackFast(bits, bitCount);
            }
        }
        bitStream.writeBitsAsText("Test.txt");
    }
    void encodePixelsT(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &channel, BitStream &bitStream) {
        for (int y = 0; y < channel.rows(); ++y) {
            for (int x = 0; x < channel.cols(); ++x) {
                T val = channel(y, x);
                auto it = codeMap.find(val);
                if (it != codeMap.end()) {
                    auto [bits, bitCount] = it->second;
                    bitStream.writeBitsBackFast(bits, bitCount);
                } else {
                    cerr << "Warning: Symbol nicht in Huffman-CodeMap\n";
                }
            }
        }
        bitStream.writeBitsAsText("Test.txt");
    }

    void printTree() {
        cout << "\n--- Huffman Tree ---\n";
        printTreeHelper(root, nullptr, false, false);
        cout << "--------------------\n";
    }

    void printTreeHex() {
        cout << "\n--- Huffman Tree ---\n";
        printTreeHelper(root, nullptr, false, true);
        cout << "--------------------\n";
    }

    void printNodesList() const
    {
        std::cout << "===== NodesList =====" << std::endl;

        for (size_t level = 0; level < nodesList.size(); ++level)
        {
            std::cout << "Level " << level << ": ";

            if (nodesList[level].empty()) {
                std::cout << "(leer)" << std::endl;
                continue;
            }

            for (Node<T>* node : nodesList[level])
            {
                // Check symbol
                std::string symbolStr;
                if (node->symbol.has_value()) {
                    symbolStr = std::to_string(*node->symbol); // as number
                } else {
                    symbolStr = "-";
                }

                std::string binaryStr;
                for (int i = node->level - 1; i >= 0; --i) {
                    binaryStr += ((node->data >> i) & 1) ? '1' : '0';
                }
                if (binaryStr.empty()) binaryStr = "0";

                std::cout << "["
                          << symbolStr
                          << ", " << binaryStr
                          << ", (" << node->frequency << ")] ";
            }
            std::cout << std::endl;
        }
        std::cout << "=====================" << std::endl;
    }


private:
    static void showTrunks(const Trunk *p) {
        if (p == nullptr) return;
        showTrunks(p->prev);
        cout << p->str;
    }

    void printTreeHelper(Node<T> *currentNode, Trunk *prev, bool isLeft, bool hex) {
        if (currentNode == nullptr) return;
        string prev_str = "    ";
        Trunk *trunk = new Trunk(prev, prev_str);
        printTreeHelper(currentNode->nodeRight, trunk, true, hex);
        if (!prev) {
            trunk->str = "---";
        } else if (isLeft) {
            trunk->str = ".---";
            prev_str = "   |";
        } else {
            trunk->str = "`---";
            prev->str = prev_str;
        }
        showTrunks(trunk);
        if (currentNode->symbol.has_value()) {
            if(!hex) {
                if constexpr (std::same_as<T, std::byte>)
                    cout << " '" << static_cast<int>(std::to_integer<unsigned char>(*currentNode->symbol)) << "':" << currentNode->frequency << endl;
                else
                    cout << " '" << *currentNode->symbol << "':" << currentNode->frequency << endl;
            }
            else {
                if constexpr (std::same_as<T, std::byte>)
                    cout << " '" << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(std::to_integer<unsigned char>(*currentNode->symbol)) <<  std::dec << "':" << currentNode->frequency << endl;
                else
                    cout << " '" << std::hex << std::setw(2) << std::setfill('0') << *currentNode->symbol <<  std::dec << "':" << currentNode->frequency << endl;
            }
        } else {
            cout << " [" << currentNode->frequency << "]" << endl;
        }
        if (prev) prev->str = prev_str;
        trunk->str = "   |";
        printTreeHelper(currentNode->nodeLeft, trunk, false, hex);
        delete trunk;
    }

    // Adds fake leave to prevent 1s only code
    void preventOneStarCode() {
        Node<T> *parentOfLastElement = getParentOfLastTreeElement(root);
        if (parentOfLastElement == nullptr) {
            return;
        }
        Node<T> *onlyOnesNode = parentOfLastElement->nodeRight;
        auto *fakeParent = new Node<T>(
            onlyOnesNode->frequency,
            onlyOnesNode->level,
            onlyOnesNode,
            nullptr
        );
        parentOfLastElement->nodeRight = fakeParent;
        onlyOnesNode->data = (onlyOnesNode->data << 1);
        onlyOnesNode->level += 1;
        codeMap[onlyOnesNode->symbol.value()] = {onlyOnesNode->data, onlyOnesNode->level};
        levels++;
    }

    Node<T> *getParentOfLastTreeElement(Node<T> *currentNode) {
        if (currentNode == nullptr) return nullptr; // Empty tree
        if (currentNode->nodeRight == nullptr) return nullptr; // Single-node tree: no parent exists
        if (currentNode->nodeRight->nodeRight == nullptr) return currentNode; // Parent of the rightmost leaf
        return getParentOfLastTreeElement(currentNode->nodeRight);
    }
};
#endif //JPEG_PROJEKT_HUFFMAN_H
