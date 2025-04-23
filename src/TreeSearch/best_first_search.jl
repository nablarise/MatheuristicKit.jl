# type is abstract because the developer may want to implement different priority rules.
abstract type AbstractBestFirstSearchStrategy <: AbstractSearchStrategy end

"""
  get_priority(strategy, space, node) -> Float

Lorem ipsum.
"""
function get_priority end

function search(strategy::AbstractBestFirstSearchStrategy, search_space)
    root_node = new_root(search_space)
    queue = PriorityQueue{typeof(root_node),Float64}()
    enqueue!(queue, root_node, get_priority(strategy, search_space, root_node))
    
    # NOTE: It's important to call `stop()` first as it may update `space`
    while !stop(search_space, queue) && !isempty(queue)
        current = dequeue!(queue)
        for child in children(search_space, current)
            enqueue!(queue, child, get_priority(strategy, search_space, child))
        end
    end
    return output(search_space, queue)
end