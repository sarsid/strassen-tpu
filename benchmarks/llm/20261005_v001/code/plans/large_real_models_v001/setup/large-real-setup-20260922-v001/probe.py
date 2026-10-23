import jax, json
items = []
for device in jax.devices():
    item = {'kind': device.device_kind, 'id': device.id, 'platform': device.platform,
            'process_index': device.process_index}
    for name in ('coords', 'core_on_chip', 'slice_index', 'local_hardware_id'):
        value = getattr(device, name, None)
        if value is not None: item[name] = value
    items.append(item)
value = {'backend': jax.default_backend(), 'device_count': len(items), 'devices': items}
print(json.dumps(value), flush=True)
assert value['backend'] == 'tpu' and len(items) == 1
assert items[0]['kind'] == 'TPU v5 lite', items[0]['kind']
