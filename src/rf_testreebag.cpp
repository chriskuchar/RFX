#include "rf_testreebag.hpp"
#include "rf_types.hpp"
#include "rf_utils.hpp"
#include <cstdio>
#include <algorithm>

namespace rf {

// ============================================================================
// CPU TESTREEBAG IMPLEMENTATION
// ============================================================================

void cpu_testreebag(const real_t* x, const real_t* xbestsplit,
                   const integer_t* nin, const integer_t* treemap,
                   const integer_t* bestvar, const integer_t* nodeclass,
                   const integer_t* cat, const integer_t* nodestatus,
                   const integer_t* catgoleft, integer_t nsample,
                   integer_t mdim, integer_t nnode, integer_t maxcat,
                   integer_t* jtr, integer_t* nodextr) {
    
    // DEBUG output disabled for production
    // printf("DEBUG: cpu_testreebag ENTRY - nsample=%d, mdim=%d, nnode=%d, maxcat=%d\n",
    //        static_cast<int>(nsample), static_cast<int>(mdim), static_cast<int>(nnode), static_cast<int>(maxcat));
    // fflush(stdout);
    
    // Initialize output arrays
    // printf("DEBUG: About to initialize output arrays\n");
    // fflush(stdout);
    zerv(jtr, nsample);
    zerv(nodextr, nsample);
    // printf("DEBUG: Output arrays initialized\n");
    // fflush(stdout);
    
    integer_t oob_count = 0;
    // printf("DEBUG: About to count OOB samples, nsample=%d\n", static_cast<int>(nsample));
    // fflush(stdout);
    
    // Safety check: verify nin pointer is valid
    if (nin == nullptr) {
        printf("ERROR: nin pointer is null!\n");
        fflush(stdout);
        return;
    }
    
    // printf("DEBUG: nin pointer is valid, starting count loop\n");
    // fflush(stdout);
    
    // Test first few accesses
    // printf("DEBUG: Testing first 10 nin values: ");
    // for (integer_t i = 0; i < std::min(10, static_cast<integer_t>(nsample)); ++i) {
    //     printf("%d ", static_cast<int>(nin[i]));
    // }
    // printf("\n");
    // fflush(stdout);
    
    for (integer_t i = 0; i < nsample; ++i) {
        // if (i % 20000 == 0 && i > 0) {
        //     printf("DEBUG: Counting OOB samples, progress: %d/%d, oob_count so far=%d\n", 
        //            static_cast<int>(i), static_cast<int>(nsample), static_cast<int>(oob_count));
        //     fflush(stdout);
        // }
        if (nin[i] == 0) oob_count++;
    }
    // printf("DEBUG: Found %d out-of-bag samples\n", static_cast<int>(oob_count));
    // fflush(stdout);
    
    // For each sample
    // printf("DEBUG: Starting sample loop, nsample=%d\n", static_cast<int>(nsample));
    // fflush(stdout);
    for (integer_t n = 0; n < nsample; ++n) {
        // if (n % 10000 == 0) {
        //     printf("DEBUG: Processing sample %d\n", static_cast<int>(n));
        //     fflush(stdout);
        // }
        if (nin[n] == 0) {  // Only out-of-bag samples
            integer_t kt = 0;  // Start at root node
            integer_t traversal_count = 0;
            const integer_t max_traversal = 1000;  // Safety limit
            
            // Traverse tree
            // Note: nodestatus values: 
            //   -1 = terminal (leaf node)
            //    1 = split (has children, can traverse)
            //    2 = non-terminal but not split yet (should not traverse)
            // Only traverse nodes that have been split (status == 1)
            while (nodestatus[kt] == 1 && traversal_count < max_traversal) {  // Only traverse split nodes
                traversal_count++;
                if (traversal_count >= max_traversal) {
                    printf("ERROR: Traversal limit reached for sample %d, node %d, status=%d\n",
                           static_cast<int>(n), static_cast<int>(kt), static_cast<int>(nodestatus[kt]));
                    fflush(stdout);
                    break;
                }
                integer_t mvar = bestvar[kt];
                
                // Safety check: if bestvar is invalid, treat as terminal
                if (mvar < 0 || mvar >= mdim) {
                    printf("WARNING: Invalid bestvar=%d for node %d, treating as terminal\n",
                           static_cast<int>(mvar), static_cast<int>(kt));
                    fflush(stdout);
                    break;
                }
                
                if (cat == nullptr || cat[mvar] == 1) {
                    // Quantitative variable (or cat is nullptr, treat as quantitative)
                    if (x[n * mdim + mvar] <= xbestsplit[kt]) {  // Fix: row-major indexing
                        kt = treemap[0 + kt * 2];  // Left child
                    } else {
                        kt = treemap[1 + kt * 2];  // Right child
                    }
                } else {
                    // Categorical variable
                    integer_t cat_value = static_cast<integer_t>(x[n * mdim + mvar]);  // Fix: row-major indexing
                    integer_t numcat = cat[mvar];
                    
                    if (cat_value >= 0 && cat_value < numcat) {
                        if (catgoleft[cat_value + kt * maxcat] == 1) {
                            kt = treemap[0 + kt * 2];  // Left child
                        } else {
                            kt = treemap[1 + kt * 2];  // Right child
                        }
                    } else {
                        // Invalid category - go to right child
                        kt = treemap[1 + kt * 2];
                    }
                }
                
                // Check bounds
                if (kt < 0 || kt >= nnode) {
                    printf("WARNING: Invalid child node %d for sample %d, treating parent as terminal\n",
                           static_cast<int>(kt), static_cast<int>(n));
                    fflush(stdout);
                    kt = 0;  // Fallback to root
                    break;
                }
                
                // Check if child node is terminal or not split
                if (nodestatus[kt] != 1) {
                    // Child is terminal (status -1) or not split (status 2), stop traversal
                    break;
                }
            }
            
            // Store results
            // If we ended on a node with status 2 (non-terminal but not split), it should still have a class
            // If status is -1 (terminal), use nodeclass directly
            if (nodestatus[kt] == -1 || nodestatus[kt] == 2) {
                jtr[n] = nodeclass[kt];
                nodextr[n] = kt;
            } else {
                // Fallback: use root node class
                printf("WARNING: Sample %d ended on node %d with unexpected status %d, using root class\n",
                       static_cast<int>(n), static_cast<int>(kt), static_cast<int>(nodestatus[kt]));
                fflush(stdout);
                jtr[n] = nodeclass[0];
                nodextr[n] = 0;
            }
        }
    }
}

} // namespace rf
