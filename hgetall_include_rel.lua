local keys, argv = KEYS, ARGV
local obj_key = table.remove(keys, 1)
local pks = argv
local index = {}
local r = {} 
local extra_keys = {}
local next = next

i = 1
while i < #keys do
    table.insert(extra_keys, {member=keys[i], fmt=keys[i + 1]})
    i = i + 2
end

for i, pk in ipairs(pks) do
    local key = obj_key:gsub('{pk}', pk)
    local obj = redis.call('hgetall', key)

    --ignore empty results
    if next(obj) ~= nil then
        table.insert(r, obj)
        
        --populate the index of keys location in the objects table, called once.
        if #index == 0 and #extra_keys > 0 then
            for i, key in ipairs(obj) do
                index[key] = i + 1
            end
        end

        --for every extra related key we want to retrieve we add it after 
        --the current item.
        for i, v in ipairs(extra_keys) do
            local key = v.fmt:gsub('{pk}', tostring(obj[index[v.member]]))
            table.insert(r, redis.call('hgetall', key))
        end
    end
end

return r
