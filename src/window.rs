use cgmath::{InnerSpace, Matrix, Matrix3, Matrix4, Point3, Rad, SquareMatrix, Vector3};
use std::{sync::Arc, time::Instant};
use vulkano::{
    buffer::{
        allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
        Buffer, BufferContents, BufferCreateInfo, BufferUsage,
    },
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo,
        QueueFlags,
    },
    format::Format,
    image::{view::ImageView, Image, ImageCreateInfo, ImageType, ImageUsage},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            depth_stencil::{DepthState, DepthStencilState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, Subpass},
    swapchain::{
        acquire_next_image, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
    },
    sync::{self, GpuFuture},
    Validated, VulkanError, VulkanLibrary,
};
use winit::{
    dpi::LogicalSize,
    event::{self, Event, MouseButton, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{CursorGrabMode, WindowBuilder},
};

// Following example from https://fby-laboratory.com/articles/article4_en
// Commented out depth buffer, reference site to re-include it (or uncomment)

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shaders/raymarch.vert"
        // src: "
        //     #version 450
        //     layout(location = 0) in vec3 position;
        //     layout(location = 1) in vec4 color;

        //     layout(set=0, binding=0)uniform MvpData{
        //         mat4 model_array;
        //         mat4 view_array;
        //         mat4 proj_array;
        //         float time;
        //     } mvpd;

        //     layout(location=0) out vec4 outcolor;

        //     void main() {
        //         //DO THESE HAVE TO BE USED?!?
        //         //vec4 mvp_position = mvpd.proj_array * mvpd.view_array * mvpd.model_array * vec4(position, 1.0);
        //         gl_Position = vec4(position, 1.0); //mvp_position;
        //         outcolor = color;
        //     }
        // "
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/raymarch.frag",
        //include: ["src/shaders",]
        // src: "
        //     #version 450
        //     layout(location=0) in vec4 incolor;

        //     layout(set=0, binding=0)uniform MvpData{
        //         mat4 model_array;
        //         mat4 view_array;
        //         mat4 proj_array;
        //         float time;
        //     } mvpd;

        //     layout(location = 0) out vec4 f_color;
        //     void main() {
        //         float test = mvpd.time;
        //         f_color = vec4(incolor.r * sin(mvpd.time), incolor.g * sin(mvpd.time), incolor.b * sin(mvpd.time), incolor.a);
        //     }
        // "
    }
}

struct Uniforms {
    model: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    time: f32,
}

pub fn main() {
    let event_loop = EventLoop::new();

    let mut input = winit_input_helper::WinitInputHelper::new();

    let library = VulkanLibrary::new().unwrap();

    let required_extensions = Surface::required_extensions(&event_loop);

    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            enabled_extensions: required_extensions,
            ..Default::default()
        },
    )
    .unwrap();

    let window = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());
    window.set_inner_size(LogicalSize::new(800, 800));
    let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();

    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    };

    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    q.queue_flags.intersects(QueueFlags::GRAPHICS)
                        && p.surface_support(i as u32, &surface).unwrap_or(false)
                })
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
            _ => 5,
        })
        .expect("No suitable physical device found");

    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type,
    );

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: device_extensions,
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
    .unwrap();

    let queue = queues.next().unwrap();

    let (mut swapchain, images) = {
        let surface_capabilities = device
            .physical_device()
            .surface_capabilities(&surface, Default::default())
            .unwrap();

        let image_format = device
            .physical_device()
            .surface_formats(&surface, Default::default())
            .unwrap()[0]
            .0;

        Swapchain::new(
            device.clone(),
            surface,
            SwapchainCreateInfo {
                min_image_count: surface_capabilities.min_image_count.max(2),
                image_format,
                image_extent: window.inner_size().into(),
                image_usage: ImageUsage::COLOR_ATTACHMENT,
                composite_alpha: surface_capabilities
                    .supported_composite_alpha
                    .into_iter()
                    .next()
                    .unwrap(),
                ..Default::default()
            },
        )
        .unwrap()
    };

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Buffers & Allocators
    ///////////////////////////////////////////////////////////////////////////////////////////////

    #[derive(BufferContents, Vertex)]
    #[repr(C)]
    struct Vertex {
        #[format(R32G32B32_SFLOAT)]
        position: [f32; 3],
        #[format(R32G32B32A32_SFLOAT)]
        color: [f32; 4],
    }

    let vertices = [
        // Full Screen Triangle Optimization (could be moved into shader and pipeline removed for 0.1 ms speed increase)
        // Vertices are placed in screen space (-1,1)
        Vertex {
            position: [-0.98, -0.98, 0.0],
            color: [1.0, 0.35, 0.137, 1.0],
        }, // top left corner
        Vertex {
            position: [-0.98, 3.0, 0.0],
            color: [1.0, 0.35, 0.137, 1.0],
        }, // bottom left corner
        Vertex {
            position: [3.0, -0.98, 0.0],
            color: [1.0, 0.35, 0.137, 1.0],
        }, // top right corner
    ];

    let vertex_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        vertices,
    )
    .unwrap();

    // let index_data: Vec<u32> = vec!(
    //     0, 1, 2, 2, 3, 0
    // );

    // let index_buffer = Buffer::from_iter(
    //     memory_allocator.clone(),
    //     BufferCreateInfo {
    //         usage: BufferUsage::INDEX_BUFFER,
    //         ..Default::default()
    //     },
    //     AllocationCreateInfo {
    //         memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
    //             | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
    //         ..Default::default()
    //     },
    //     index_data,
    // )
    // .unwrap();

    // let depth_buffer = ImageView::new_default(
    //     Image::new(
    //         memory_allocator.clone(),
    //         ImageCreateInfo {
    //             image_type: ImageType::Dim2d,
    //             format: Format::D16_UNORM,
    //             extent: images[0].extent(),
    //             usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::TRANSIENT_ATTACHMENT,
    //             ..Default::default()
    //         },
    //         AllocationCreateInfo::default(),
    //     )
    //     .unwrap(),
    // )
    // .unwrap();

    let uniform_buffer = SubbufferAllocator::new(
        memory_allocator.clone(),
        SubbufferAllocatorCreateInfo {
            buffer_usage: BufferUsage::UNIFORM_BUFFER,
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
    );

    let uniform_buffer_subbuffer = {

        let uniform_data = vs::MvpData {
            model: Default::default(),
            view: Default::default(),
            proj: Default::default(),
            time: 0 as f32,
        };

        let subbuffer = uniform_buffer.allocate_sized().unwrap();
        *subbuffer.write().unwrap() = uniform_data;

        subbuffer
    };

    let vs = vs::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();
    let fs = fs::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();

    let render_pass = vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                format: swapchain.image_format(),
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },
            // depth: {
            //     format: Format::D16_UNORM,
            //     samples: 1,
            //     load_op: Clear,
            //     store_op: DontCare,
            // }
        },
        pass: {
            color: [color],
            depth_stencil: {} //depth
        }
    )
    .unwrap();

    let vertex_input_state = Vertex::per_vertex()
        .definition(&vs.info().input_interface)
        .unwrap();

    let stages = [
        PipelineShaderStageCreateInfo::new(vs.clone()),
        PipelineShaderStageCreateInfo::new(fs.clone()),
    ]
    .into_iter()
    .collect();

    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )
    .unwrap();

    let mut pipeline = GraphicsPipeline::new(
        device.clone(),
        None,
        GraphicsPipelineCreateInfo {
            stages,
            vertex_input_state: Some(vertex_input_state),
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(ViewportState::default()),
            rasterization_state: Some(RasterizationState::default()),
            // depth_stencil_state: Some(DepthStencilState {
            //     depth: Some(DepthState::simple()),
            //     ..Default::default()
            // }),
            multisample_state: Some(MultisampleState::default()),
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                Subpass::from(render_pass.clone(), 0)
                    .unwrap()
                    .num_color_attachments(),
                ColorBlendAttachmentState::default(),
            )),
            dynamic_state: [DynamicState::Viewport].into_iter().collect(),
            subpass: Some(Subpass::from(render_pass.clone(), 0).unwrap().into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .unwrap();

    let mut viewport = Viewport {
        offset: [0.0, 0.0],
        extent: [0.0, 0.0],
        depth_range: 0.0..=1.0,
    };

    //let extent = images[0].extent();
    viewport.extent = window.inner_size().into(); //[extent[0] as f32, extent[1] as f32];

    let mut framebuffers = images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![
                        view,
                        //depth_buffer.clone()
                    ],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>();

    let descriptor_set_allocator =
        StandardDescriptorSetAllocator::new(device.clone(), Default::default());

    let command_buffer_allocator =
        StandardCommandBufferAllocator::new(device.clone(), Default::default());

    let mut recreate_swapchain = false;

    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

    // Begin timer
    let instant = Instant::now();
    let mut elapsed = instant.elapsed();

    // try confining the cursor, and if that fails, try locking it instead.
    // window.set_cursor_grab(CursorGrabMode::Confined)
    //         .or_else(|_e| window.set_cursor_grab(CursorGrabMode::Locked))
    //         .unwrap();

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Main Event Loop
    ///////////////////////////////////////////////////////////////////////////////////////////////
    event_loop.run(move |event, _, control_flow| {
        //*control_flow = ControlFlow::Poll;

        // Update input
        //crate::input::update(&mut input, &event); // not sure if theres a better way to structure files, this checks the top level crate
        if input.update(&event){
            if input.key_pressed(winit::event::VirtualKeyCode::W) {
                println!("The 'W' key (US layout) was pressed on the keyboard");
            }
    
            if input.key_held(winit::event::VirtualKeyCode::R) {
                println!("The 'R' key (US layout) key is held");
            }

            if input.key_held(event::VirtualKeyCode::LAlt){
                // cursor lock or hide or somthn idk
            }
    
            // query the change in cursor this update
            if input.mouse_held(0) {
                let cursor_diff = input.mouse_diff();
                if cursor_diff != (0.0, 0.0) {
                    println!("The cursor diff is: {:?}", cursor_diff);
                    println!("The cursor position is: {:?}", input.mouse()); // Return mouse coordinates in pixels
                }
            }
        }
        
        // Main render loop
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                recreate_swapchain = true;
            }
            Event::MainEventsCleared => { //RedrawEventsCleared
                let image_extent: [u32; 2] = window.inner_size().into();
                if image_extent.contains(&0) {
                    return;
                }

                previous_frame_end.as_mut().unwrap().cleanup_finished();

                if recreate_swapchain {
                    let (new_swapchain, new_images) = swapchain
                        .recreate(SwapchainCreateInfo {
                            image_extent,
                            ..swapchain.create_info()
                        })
                        .expect("failed to recreate swapchain");

                    swapchain = new_swapchain;

                    framebuffers = new_images
                        .iter()
                        .map(|image| {
                            let view = ImageView::new_default(image.clone()).unwrap();
                            Framebuffer::new(
                                render_pass.clone(),
                                FramebufferCreateInfo {
                                    attachments: vec![
                                        view,
                                        //depth_buffer.clone()
                                    ],
                                    ..Default::default()
                                },
                            )
                            .unwrap()
                        })
                        .collect::<Vec<_>>();

                    // Resizes window on re-create (can do without) ///////////////////////////////////////////////////
                    let vertex_input_state = Vertex::per_vertex()
                        .definition(&vs.info().input_interface)
                        .unwrap();

                    let stages = [
                        PipelineShaderStageCreateInfo::new(vs.clone()),
                        PipelineShaderStageCreateInfo::new(fs.clone()),
                    ]
                    .into_iter()
                    .collect();

                    let layout = PipelineLayout::new(
                        device.clone(),
                        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                            .into_pipeline_layout_create_info(device.clone())
                            .unwrap(),
                    )
                    .unwrap();

                    viewport.extent = window.inner_size().into();
                    pipeline = GraphicsPipeline::new(
                        device.clone(),
                        None,
                        GraphicsPipelineCreateInfo {
                            stages,
                            vertex_input_state: Some(vertex_input_state),
                            input_assembly_state: Some(InputAssemblyState::default()),
                            viewport_state: Some(ViewportState {
                                viewports: [viewport.clone()].into_iter().collect(),
                                ..Default::default()
                            }),
                            rasterization_state: Some(RasterizationState::default()),
                            // depth_stencil_state: Some(DepthStencilState {
                            //     depth: Some(DepthState::simple()),
                            //     ..Default::default()
                            // }),
                            multisample_state: Some(MultisampleState::default()),
                            color_blend_state: Some(ColorBlendState::with_attachment_states(
                                Subpass::from(render_pass.clone(), 0)
                                    .unwrap()
                                    .num_color_attachments(),
                                ColorBlendAttachmentState::default(),
                            )),
                            dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                            subpass: Some(Subpass::from(render_pass.clone(), 0).unwrap().into()),
                            ..GraphicsPipelineCreateInfo::layout(layout)
                        },
                    )
                    .unwrap();
                    ///////////////////////////////////////////////////////////////////////////////////////////////////


                    recreate_swapchain = false;
                }

                ///////////////////////////////////////////////////////////////////////////////////
                // Update Uniforms & Descriptor
                ///////////////////////////////////////////////////////////////////////////////////
                let uniform_buffer_subbuffer = {
                    let mut x_thita = 0.0 ;
                    let y_thita = 0.0;
                    let z_thita = 0.0;

                    x_thita += input.mouse_diff().0;

                    let nx = Vector3::new(1.0, 0.0, 0.0);
                    let rotation_x = Matrix3::from_axis_angle(nx, Rad(x_thita));

                    let ny = Vector3::new(0.0, 1.0, 0.0);
                    let rotation_y = Matrix3::from_axis_angle(ny, Rad(y_thita));

                    let nz = Vector3::new(0.0, 0.0, 1.0);
                    let rotation_z = Matrix3::from_axis_angle(nz, Rad(z_thita));

                    let rotation = Matrix4::from(rotation_x * rotation_y * rotation_z);

                    let x_translation = 0.0;
                    let y_translation = 0.0;
                    let z_translation = 0.0;

                    let translation = Matrix4::from_translation(Vector3::new(
                        x_translation,
                        y_translation,
                        z_translation,
                    ));

                    let x_scale = 1.0;
                    let y_scale = 1.0;
                    let z_scale = 1.0;

                    let scale = Matrix4::from_nonuniform_scale(x_scale, y_scale, z_scale);

                    let model = translation * rotation * scale;

                    let model_array = [
                        [model.x.x, model.x.y, model.x.z, model.x.w],
                        [model.y.x, model.y.y, model.y.z, model.y.w],
                        [model.z.x, model.z.y, model.z.z, model.z.w],
                        [model.w.x, model.w.y, model.w.z, model.w.w],
                    ];

                    let eye_position = Point3::new(0.0, 1.0, 1.0);
                    let looking_point = Point3::new(0.0, 0.0, 0.0);

                    let looking_dir = looking_point - eye_position;
                    let unit_z = Vector3::new(0.0, 0.0, 1.0);
                    let e = unit_z.cross(-looking_dir);
                    let up_direction = e.cross(looking_dir).normalize();

                    let view = Matrix4::look_at_rh(eye_position, looking_point, up_direction);

                    let view_array = [
                        [view.x.x, view.x.y, view.x.z, view.x.w],
                        [view.y.x, view.y.y, view.y.z, view.y.w],
                        [view.z.x, view.z.y, view.z.z, view.z.w],
                        [view.w.x, view.w.y, view.w.z, view.w.w],
                    ];

                    let aspect_ratio =
                        swapchain.image_extent()[0] as f32 / swapchain.image_extent()[1] as f32;

                    let mut proj = cgmath::perspective(
                        Rad(std::f32::consts::FRAC_PI_2),
                        aspect_ratio,
                        0.01,
                        100.0,
                    );

                    proj.x *= -1.0;
                    proj.y *= -1.0;

                    let proj_array = [
                        [proj.x.x, proj.x.y, proj.x.z, proj.x.w],
                        [proj.y.x, proj.y.y, proj.y.z, proj.y.w],
                        [proj.z.x, proj.z.y, proj.z.z, proj.z.w],
                        [proj.w.x, proj.w.y, proj.w.z, proj.w.w],
                    ];

                    let uniform_data = vs::MvpData {
                        model: model_array,
                        view: view_array,
                        proj: proj_array,
                        time: instant.elapsed().as_secs_f32(),
                    };

                    let subbuffer = uniform_buffer.allocate_sized().unwrap();
                    *subbuffer.write().unwrap() = uniform_data;

                    subbuffer
                };
                

                let descriptor_layout = pipeline.layout().set_layouts().get(0).unwrap();
                let descriptor_set = PersistentDescriptorSet::new(
                    &descriptor_set_allocator,
                    descriptor_layout.clone(),
                    [WriteDescriptorSet::buffer(
                        0,
                        uniform_buffer_subbuffer.clone(),
                    )],
                    [],
                )
                .unwrap();

                let (image_index, suboptimal, acquire_future) =
                    match acquire_next_image(swapchain.clone(), None).map_err(Validated::unwrap) {
                        Ok(r) => r,
                        Err(VulkanError::OutOfDate) => {
                            recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("failed to acquire next image: {e}"),
                    };

                if suboptimal {
                    recreate_swapchain = true;
                }

                ////////////////////////////////////////////////////////////////////////////////
                // Update Command Buffers
                ////////////////////////////////////////////////////////////////////////////////
                let mut builder = AutoCommandBufferBuilder::primary(
                    &command_buffer_allocator,
                    queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                builder
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            clear_values: vec![
                                Some([0.0, 0.0, 1.0, 1.0].into()), // Default draw color, no vertices
                                //Some(1f32.into()),
                            ],
                            ..RenderPassBeginInfo::framebuffer(
                                framebuffers[image_index as usize].clone(),
                            )
                        },
                        SubpassBeginInfo {
                            contents: SubpassContents::Inline,
                            ..Default::default()
                        },
                    )
                    .unwrap()
                    .set_viewport(0, [viewport.clone()].into_iter().collect())
                    .unwrap()
                    .bind_pipeline_graphics(pipeline.clone())
                    .unwrap()
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        pipeline.layout().clone(),
                        0,
                        descriptor_set,
                    )
                    .unwrap()
                    .bind_vertex_buffers(0, vertex_buffer.clone())
                    .unwrap()
                    .draw(vertex_buffer.len() as u32, 1, 0, 0)
                    .unwrap()
                    // .bind_index_buffer(index_buffer.clone())
                    // .unwrap()
                    // .draw_indexed(index_buffer.len() as u32, 1, 0, 0, 0) // IMPORTANT! Either use .draw (above) or .draw_indexed, DO NOT USE BOTH
                    // .unwrap()
                    .end_render_pass(Default::default())
                    .unwrap();

                let command_buffer = builder.build().unwrap();

                ///////////////////////////////////////////////////////////////////////////////////
                // Fences & Futures
                ///////////////////////////////////////////////////////////////////////////////////
                let future = previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(
                        queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_index),
                    )
                    .then_signal_fence_and_flush();

                match future.map_err(Validated::unwrap) {
                    Ok(future) => {
                        previous_frame_end = Some(future.boxed());
                    }
                    Err(VulkanError::OutOfDate) => {
                        recreate_swapchain = true;
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                    Err(e) => {
                        panic!("failed to flush future: {e}");
                    }
                }
            }
            _ => (),
        }
    });
}

// fn fr(){
//     let example: Mat4 = glam::Mat4::look_at_rh(eye, center, up);
//     cgmath::Matrix4::look_at_rh(eye, center, up)
// }
